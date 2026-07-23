"""NOTEBOOK_08_METADATA_GENERATION_FRESH_V1

Generate accurate, engagement-oriented publication metadata from a checked
Educational Shorts script and its fact-check report.

This module does not add facts to the video. The language model creates a
metadata draft, while Python normalizes and validates platform constraints.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from educational_shorts.client import ask_llm
from educational_shorts.schemas import FactCheckReport, VideoScript


class MetadataDraft(BaseModel):
    """Creative metadata returned by the local language model."""

    title: str = Field(
        min_length=1,
        description="The primary public-facing video title.",
    )

    alternate_titles: list[str] = Field(
        default_factory=list,
        description="Alternative accurate title options.",
    )

    short_caption: str = Field(
        min_length=1,
        description="A short caption suitable for a short-form video post.",
    )

    description: str = Field(
        min_length=1,
        description="A concise educational description of the video.",
    )

    thumbnail_text: str = Field(
        min_length=1,
        description="Very short text suitable for a thumbnail or opening card.",
    )

    hashtags: list[str] = Field(
        default_factory=list,
        description="Relevant public-facing hashtags.",
    )

    keywords: list[str] = Field(
        default_factory=list,
        description="Search and indexing keywords without hashtag symbols.",
    )

    category: str = Field(
        min_length=1,
        description="A broad content category such as Science or History.",
    )

    audience_level: Literal["general", "intermediate", "advanced"] = Field(
        description="The assumed background knowledge of the intended viewer.",
    )


class VideoMetadata(BaseModel):
    """Validated metadata plus deterministic pipeline traceability fields."""

    title: str
    alternate_titles: list[str]
    short_caption: str
    description: str
    thumbnail_text: str
    hashtags: list[str]
    keywords: list[str]
    category: str
    audience_level: Literal["general", "intermediate", "advanced"]

    filename_slug: str
    source_topic_title: str
    source_script_filename: str
    fact_check_report_filename: str

    script_word_count: int
    estimated_seconds: int

    fact_check_verdict: Literal["pass", "revised", "manual_review"]
    requires_manual_review: bool
    fact_check_summary: str

    source_urls: list[str] = Field(default_factory=list)
    generated_at_utc: str


def slugify(value: str) -> str:
    """Convert text into a stable lowercase filename slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "video"


def find_checked_script_file(
    scripts_directory: Path,
    filename: str | None = None,
) -> Path:
    """Use an explicit checked script or the newest checked-script JSON."""
    if filename:
        path = scripts_directory / filename

        if not path.exists():
            raise FileNotFoundError(f"Checked script not found: {path}")

        return path

    candidates = sorted(
        scripts_directory.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No checked script JSON files found in {scripts_directory}."
        )

    return candidates[0]


def load_script(path: Path) -> VideoScript:
    """Load and validate a checked script."""
    return VideoScript.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def find_fact_check_report_file(
    reports_directory: Path,
    script: VideoScript,
    filename: str | None = None,
) -> Path:
    """Resolve the fact-check report matching the checked script."""
    if filename:
        path = reports_directory / filename

        if not path.exists():
            raise FileNotFoundError(f"Fact-check report not found: {path}")

        return path

    expected = (
        reports_directory
        / f"{slugify(script.topic.title)}_fact_check.json"
    )

    if expected.exists():
        return expected

    raise FileNotFoundError(
        "Could not find the matching fact-check report. Expected: "
        f"{expected}. Set FACT_CHECK_REPORT_FILENAME explicitly if the "
        "report uses a different filename."
    )


def load_fact_check_report(path: Path) -> FactCheckReport:
    """Load and validate a retrieval-grounded fact-check report."""
    return FactCheckReport.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def collect_source_urls(report: FactCheckReport) -> list[str]:
    """Collect only URLs referenced by verification results."""
    urls: list[str] = []

    for verification in report.verifications:
        if verification.evidence_url:
            urls.append(verification.evidence_url)

        urls.extend(verification.supporting_urls)

    seen: set[str] = set()
    unique: list[str] = []

    for url in urls:
        normalized = url.strip()

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        unique.append(normalized)

    return unique


def build_metadata_prompt(
    script: VideoScript,
    report: FactCheckReport,
) -> str:
    """Create the metadata request without sending the full evidence archive."""
    script_payload = {
        "topic": script.topic.model_dump(),
        "hook": script.hook.narration,
        "sections": [
            {
                "segment_type": section.segment_type,
                "narration": section.narration,
            }
            for section in script.sections
        ],
        "closing": script.closing.narration,
        "full_narration": script.full_narration,
        "word_count": script.word_count,
        "estimated_total_seconds": script.estimated_total_seconds,
    }

    fact_check_payload = {
        "verdict": report.verdict,
        "requires_manual_review": report.requires_manual_review,
        "summary": report.summary,
        "corrected_claims": [
            {
                "verdict": verification.verdict,
                "explanation": verification.explanation,
                "correction": verification.correction,
            }
            for verification in report.verifications
            if verification.verdict != "supported"
        ],
    }

    return f"""
Create publication metadata for this checked short-form educational video.

CHECKED SCRIPT:
{json.dumps(script_payload, indent=2, ensure_ascii=False)}

FACT-CHECK STATUS:
{json.dumps(fact_check_payload, indent=2, ensure_ascii=False)}

Requirements:
- Use only ideas already present in the checked script.
- Do not reintroduce claims that the fact checker removed, softened, or marked
  unsupported.
- Do not add dates, numbers, names, mechanisms, outcomes, or promises that are
  absent from the checked script.
- Make the primary title accurate, curiosity-driven, and no more than 80
  characters.
- Provide 3 to 5 distinct alternate titles, each no more than 80 characters.
- The short caption must be 40 to 220 characters.
- The description must be 80 to 600 characters and use one or two short
  paragraphs.
- Thumbnail text must be 2 to 6 words and no more than 32 characters.
- Provide 5 to 8 specific, relevant hashtags.
- Provide 8 to 15 useful search keywords without hashtag symbols.
- Avoid fake urgency, unsupported superlatives, sensationalism, all caps,
  emojis, URLs, and claims of certainty beyond the checked script.
- Do not mention sources, evidence, fact checking, uncertainty in the research
  process, or manual review in the public-facing metadata.
- The metadata should sound natural rather than keyword-stuffed.
- Return only structured output matching the requested schema.
""".strip()


def _clean_single_line(text: str) -> str:
    return " ".join(text.split()).strip()


def _clean_description(text: str) -> str:
    """Preserve at most one paragraph break while cleaning whitespace."""
    paragraphs = [
        " ".join(paragraph.split())
        for paragraph in re.split(r"\n\s*\n", text.strip())
        if paragraph.strip()
    ]

    return "\n\n".join(paragraphs[:2])


def _normalize_hashtag(tag: str) -> str:
    """Create a valid hashtag while preserving useful capitalization."""
    base = tag.strip().lstrip("#")
    words = re.findall(r"[A-Za-z0-9]+", base)

    if not words:
        return ""

    if len(words) == 1:
        body = words[0]
    else:
        body = "".join(word[:1].upper() + word[1:] for word in words)

    return f"#{body}"


def _deduplicate(
    values: list[str],
    *,
    case_sensitive: bool = False,
) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned = _clean_single_line(value)

        if not cleaned:
            continue

        key = cleaned if case_sensitive else cleaned.casefold()

        if key in seen:
            continue

        seen.add(key)
        output.append(cleaned)

    return output


def normalize_metadata_draft(
    draft: MetadataDraft,
) -> MetadataDraft:
    """Normalize formatting before deterministic validation."""
    draft.title = _clean_single_line(draft.title)
    draft.alternate_titles = _deduplicate(
        [_clean_single_line(title) for title in draft.alternate_titles]
    )
    draft.short_caption = _clean_single_line(draft.short_caption)
    draft.description = _clean_description(draft.description)
    draft.thumbnail_text = _clean_single_line(draft.thumbnail_text)

    hashtags = [
        _normalize_hashtag(tag)
        for tag in draft.hashtags
    ]
    draft.hashtags = _deduplicate(hashtags)

    keywords = [
        keyword.strip().lstrip("#")
        for keyword in draft.keywords
    ]
    draft.keywords = _deduplicate(keywords)

    draft.category = _clean_single_line(draft.category)

    return draft


def validate_metadata_draft(
    draft: MetadataDraft,
) -> list[str]:
    """Return human-readable validation failures for LLM retry feedback."""
    failures: list[str] = []

    if not 8 <= len(draft.title) <= 80:
        failures.append(
            f"primary title must be 8-80 characters; received {len(draft.title)}"
        )

    if not 3 <= len(draft.alternate_titles) <= 5:
        failures.append(
            "alternate_titles must contain 3-5 distinct options; received "
            f"{len(draft.alternate_titles)}"
        )

    long_alternates = [
        title
        for title in draft.alternate_titles
        if len(title) > 80
    ]

    if long_alternates:
        failures.append(
            "every alternate title must be no more than 80 characters"
        )

    title_keys = {
        draft.title.casefold(),
        *(title.casefold() for title in draft.alternate_titles),
    }

    if len(title_keys) != len(draft.alternate_titles) + 1:
        failures.append(
            "the primary and alternate titles must all be distinct"
        )

    if not 40 <= len(draft.short_caption) <= 220:
        failures.append(
            "short_caption must be 40-220 characters; received "
            f"{len(draft.short_caption)}"
        )

    if not 80 <= len(draft.description) <= 600:
        failures.append(
            "description must be 80-600 characters; received "
            f"{len(draft.description)}"
        )

    if not 2 <= len(draft.thumbnail_text) <= 32:
        failures.append(
            "thumbnail_text must be 2-32 characters; received "
            f"{len(draft.thumbnail_text)}"
        )

    thumbnail_words = draft.thumbnail_text.split()

    if not 2 <= len(thumbnail_words) <= 6:
        failures.append(
            "thumbnail_text must contain 2-6 words; received "
            f"{len(thumbnail_words)}"
        )

    if not 5 <= len(draft.hashtags) <= 8:
        failures.append(
            "hashtags must contain 5-8 distinct values; received "
            f"{len(draft.hashtags)}"
        )

    if any(
        not re.fullmatch(r"#[A-Za-z0-9]+", hashtag)
        for hashtag in draft.hashtags
    ):
        failures.append(
            "each hashtag must start with # and contain only letters or numbers"
        )

    if not 8 <= len(draft.keywords) <= 15:
        failures.append(
            "keywords must contain 8-15 distinct values; received "
            f"{len(draft.keywords)}"
        )

    if any("#" in keyword for keyword in draft.keywords):
        failures.append("keywords must not contain hashtag symbols")

    public_text = "\n".join(
        [
            draft.title,
            *draft.alternate_titles,
            draft.short_caption,
            draft.description,
            draft.thumbnail_text,
        ]
    )

    if re.search(r"https?://|www\.", public_text, flags=re.IGNORECASE):
        failures.append("public-facing metadata must not contain URLs")

    forbidden_process_terms = re.compile(
        r"\b(the sources?|the evidence|fact[- ]?check(?:er|ing)?|"
        r"manual review|retrieved passages?)\b",
        flags=re.IGNORECASE,
    )

    if forbidden_process_terms.search(public_text):
        failures.append(
            "public-facing metadata must not discuss sources or fact checking"
        )

    return failures


def generate_metadata(
    script: VideoScript,
    report: FactCheckReport,
    system_prompt: str,
    source_script_filename: str,
    fact_check_report_filename: str,
    temperature: float = 0.5,
    seed: int | None = None,
    max_attempts: int = 4,
) -> VideoMetadata:
    """Generate, normalize, validate, and finalize metadata."""
    base_prompt = build_metadata_prompt(script, report)
    attempt_prompt = base_prompt
    failures_history: list[str] = []

    for attempt in range(1, max_attempts + 1):
        attempt_seed = None if seed is None else seed + attempt - 1

        draft = ask_llm(
            system_prompt=system_prompt,
            user_prompt=attempt_prompt,
            schema=MetadataDraft,
            temperature=temperature,
            seed=attempt_seed,
        )

        draft = normalize_metadata_draft(draft)
        failures = validate_metadata_draft(draft)

        if not failures:
            return VideoMetadata(
                title=draft.title,
                alternate_titles=draft.alternate_titles,
                short_caption=draft.short_caption,
                description=draft.description,
                thumbnail_text=draft.thumbnail_text,
                hashtags=draft.hashtags,
                keywords=draft.keywords,
                category=draft.category,
                audience_level=draft.audience_level,
                filename_slug=slugify(draft.title),
                source_topic_title=script.topic.title,
                source_script_filename=source_script_filename,
                fact_check_report_filename=fact_check_report_filename,
                script_word_count=script.word_count,
                estimated_seconds=script.estimated_total_seconds,
                fact_check_verdict=report.verdict,
                requires_manual_review=report.requires_manual_review,
                fact_check_summary=report.summary,
                source_urls=collect_source_urls(report),
                generated_at_utc=datetime.now(timezone.utc).isoformat(),
            )

        failure_text = "; ".join(failures)
        failures_history.append(
            f"attempt {attempt}: {failure_text}"
        )

        attempt_prompt = f"""
{base_prompt}

RETRY FEEDBACK:
The previous metadata draft failed these deterministic checks:
- {chr(10).join(failures)}

Return a new complete metadata draft that fixes every listed problem while
remaining faithful to the checked script.
""".strip()

    raise ValueError(
        "Could not generate valid metadata after "
        f"{max_attempts} attempts. "
        + " | ".join(failures_history)
    )


def build_metadata_filename(metadata: VideoMetadata) -> str:
    return f"{metadata.filename_slug}.json"


def save_metadata(
    metadata: VideoMetadata,
    output_path: Path,
) -> None:
    """Save formatted UTF-8 metadata JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            metadata.model_dump(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def summarize_metadata(
    metadata: VideoMetadata,
) -> dict[str, object]:
    """Return a compact notebook summary."""
    return {
        "title": metadata.title,
        "category": metadata.category,
        "audience_level": metadata.audience_level,
        "hashtags": len(metadata.hashtags),
        "keywords": len(metadata.keywords),
        "estimated_seconds": metadata.estimated_seconds,
        "fact_check_verdict": metadata.fact_check_verdict,
        "requires_manual_review": metadata.requires_manual_review,
        "source_urls": len(metadata.source_urls),
    }
