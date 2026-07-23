from __future__ import annotations

import json
import re
from pathlib import Path

from educational_shorts.client import ask_llm
from educational_shorts.schemas import ScriptSegment, VideoOutline, VideoScript


def find_outline_file(
    outlines_directory: Path,
    filename: str | None = None,
) -> Path:
    """Resolve an outline JSON file explicitly or choose the most recent one."""
    if filename is not None:
        path = outlines_directory / filename
        if not path.exists():
            raise FileNotFoundError(f"Outline file not found: {path}")
        return path

    candidates = sorted(
        outlines_directory.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No outline JSON files were found in {outlines_directory}."
        )

    return candidates[0]


def build_script_prompt(
    outline: VideoOutline,
    target_wpm: int = 145,
) -> str:
    """Build the user prompt for structured narration generation."""
    if target_wpm < 80:
        raise ValueError("target_wpm must be at least 80.")

    outline_json = json.dumps(
        outline.model_dump(),
        indent=2,
        ensure_ascii=False,
    )
    approximate_word_target = round(
        outline.estimated_total_seconds * target_wpm / 60
    )

    return f"""
Write the complete spoken narration for the educational short represented by
this outline:

{outline_json}

Requirements:
- Preserve the exact topic and teaching objective.
- Write one hook segment, exactly {len(outline.sections)} body segments, and
  one closing segment.
- Keep every body segment aligned with the corresponding outline section.
- Use natural spoken English for a general audience.
- Explain technical terms immediately and simply.
- Avoid repetitive setup, filler, stage directions, citations, and greetings.
- Do not invent statistics, studies, quotations, or claims absent from the
  outline.
- Assume a speaking rate near {target_wpm} words per minute.
- Aim for roughly {approximate_word_target} total words so the narration fits
  approximately {outline.estimated_total_seconds} seconds.
- Keep the hook brief and end with one clear takeaway.
- Put visual guidance in visual_direction, not inside narration.
- Return only valid structured output matching the requested schema.
""".strip()


def count_words(text: str) -> int:
    """Count ordinary spoken words in a block of narration."""
    return len(re.findall(r"\b[\w’'-]+\b", text))


def estimate_seconds(text: str, words_per_minute: int = 145) -> int:
    """Estimate spoken duration from narration word count."""
    if words_per_minute < 1:
        raise ValueError("words_per_minute must be at least 1.")
    return max(1, round(count_words(text) * 60 / words_per_minute))


def assemble_full_narration(script: VideoScript) -> str:
    """Join all narration segments in their intended spoken order."""
    parts = [
        script.hook.narration.strip(),
        *(section.narration.strip() for section in script.sections),
        script.closing.narration.strip(),
    ]
    return "\n\n".join(part for part in parts if part)


def _normalize_segment_timing(
    segment: ScriptSegment,
    words_per_minute: int,
) -> None:
    segment.estimated_seconds = estimate_seconds(
        segment.narration,
        words_per_minute=words_per_minute,
    )


def generate_script(
    outline: VideoOutline,
    system_prompt: str,
    target_wpm: int = 145,
    temperature: float = 0.5,
    seed: int | None = None,
) -> VideoScript:
    """Generate, normalize, and validate narration for one video outline."""
    script = ask_llm(
        system_prompt=system_prompt,
        user_prompt=build_script_prompt(outline, target_wpm),
        schema=VideoScript,
        temperature=temperature,
        seed=seed,
    )

    if len(script.sections) != len(outline.sections):
        raise ValueError(
            "The generated script does not match the outline: "
            f"expected {len(outline.sections)} body sections but received "
            f"{len(script.sections)}."
        )

    script.topic = outline.topic

    for script_section, outline_section in zip(
        script.sections,
        outline.sections,
        strict=True,
    ):
        if not script_section.visual_direction.strip():
            script_section.visual_direction = outline_section.visual_direction

    _normalize_segment_timing(script.hook, target_wpm)
    for section in script.sections:
        _normalize_segment_timing(section, target_wpm)
    _normalize_segment_timing(script.closing, target_wpm)

    script.full_narration = assemble_full_narration(script)
    script.word_count = count_words(script.full_narration)
    script.estimated_total_seconds = sum(
        [
            script.hook.estimated_seconds,
            *(section.estimated_seconds for section in script.sections),
            script.closing.estimated_seconds,
        ]
    )

    return script


def slugify(value: str) -> str:
    """Convert text into a filesystem-friendly lowercase slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "script"


def build_script_filename(outline: VideoOutline) -> str:
    """Create a stable JSON filename from the outline topic."""
    return f"{slugify(outline.topic.title)}.json"


def save_script(script: VideoScript, output_path: Path) -> None:
    """Save a validated video script as formatted UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(script.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_script(input_path: Path) -> VideoScript:
    """Load and validate a previously saved video script."""
    return VideoScript.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )