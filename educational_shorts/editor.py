from __future__ import annotations

import json
import re
from pathlib import Path

from educational_shorts.client import ask_llm
from educational_shorts.schemas import ScriptSegment, VideoScript


def find_script_file(
    scripts_directory: Path,
    filename: str | None = None,
) -> Path:
    """Resolve a script JSON file explicitly or choose the newest one."""
    if filename is not None:
        path = scripts_directory / filename

        if not path.exists():
            raise FileNotFoundError(f"Script file not found: {path}")

        return path

    candidates = sorted(
        scripts_directory.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No script JSON files were found in {scripts_directory}."
        )

    return candidates[0]


def load_script(input_path: Path) -> VideoScript:
    """Load and validate a saved script.

    Older JSON files may contain visual_direction fields. Pydantic ignores
    those extra fields, so they do not interfere with the simplified schema.
    """
    return VideoScript.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )


def count_words(text: str) -> int:
    """Count ordinary spoken words."""
    return len(re.findall(r"\b[\w’'-]+\b", text))


def estimate_seconds(
    text: str,
    words_per_minute: int = 145,
) -> int:
    """Estimate narration duration from its word count."""
    if words_per_minute < 1:
        raise ValueError("words_per_minute must be at least 1.")

    return max(
        1,
        round(count_words(text) * 60 / words_per_minute),
    )


def assemble_full_narration(script: VideoScript) -> str:
    """Join all spoken segments in playback order."""
    parts = [
        script.hook.narration.strip(),
        *(section.narration.strip() for section in script.sections),
        script.closing.narration.strip(),
    ]

    return "\n\n".join(part for part in parts if part)


def normalize_script(
    script: VideoScript,
    words_per_minute: int = 145,
) -> VideoScript:
    """Recalculate all timing and summary fields from the narration."""
    segments = [
        script.hook,
        *script.sections,
        script.closing,
    ]

    for segment in segments:
        segment.narration = " ".join(segment.narration.split())
        segment.estimated_seconds = estimate_seconds(
            segment.narration,
            words_per_minute=words_per_minute,
        )

    script.full_narration = assemble_full_narration(script)
    script.word_count = count_words(script.full_narration)
    script.estimated_total_seconds = sum(
        segment.estimated_seconds for segment in segments
    )

    return script


def build_editor_prompt(
    script: VideoScript,
    target_wpm: int = 145,
    minimum_seconds: int = 40,
    maximum_seconds: int = 60,
) -> str:
    """Build the editing request for the local language model."""
    if minimum_seconds < 1:
        raise ValueError("minimum_seconds must be at least 1.")

    if maximum_seconds < minimum_seconds:
        raise ValueError(
            "maximum_seconds must be greater than or equal to minimum_seconds."
        )

    minimum_words = 90
    maximum_words = round(maximum_seconds * target_wpm / 60)

    script_json = json.dumps(
        script.model_dump(),
        indent=2,
        ensure_ascii=False,
    )

    return f"""
Edit the short-form narration below for stronger viewer retention while
preserving its core topic and factual meaning.

SOURCE SCRIPT:
{script_json}

Editing requirements:
- Return one hook, exactly {len(script.sections)} body sections, and one closing.
- Preserve the original topic and the order of the main ideas.
- Do not add statistics, studies, quotations, names, or factual claims that are
  not already supported by the source script.
- Improve clarity and precision without turning the script into a lecture.
- Remove repeated ideas, filler, greetings, and unnecessary setup.
- Use short, natural sentences that sound good when spoken aloud.
- Prefer active voice and concrete wording.
- Explain unavoidable technical terms immediately in plain language.
- Make the hook create immediate curiosity. Do not begin with "Did you know".
- Keep the hook brief enough to deliver in about 3 to 6 seconds.
- Give each body section a distinct job and a smooth transition from the
  previous section.
- The closing should deliver a payoff or memorable takeaway rather than merely
  repeating the introduction.
- Do not include visual directions, stage directions, hashtags, citations,
  headings, or calls to like, follow, or subscribe inside narration.
- Aim for {minimum_words} to {maximum_words} total words, corresponding to
  approximately {minimum_seconds} to {maximum_seconds} seconds at
  {target_wpm} words per minute.
- Return only valid structured output matching the requested schema.
""".strip()


def edit_script(
    script: VideoScript,
    system_prompt: str,
    target_wpm: int = 145,
    minimum_seconds: int = 40,
    maximum_seconds: int = 60,
    temperature: float = 0.4,
    seed: int | None = None,
    minimum_words: int = 90,
    max_attempts: int = 3,
) -> VideoScript:
    """Edit a script and retry when the result is too short."""

    for attempt in range(1, max_attempts + 1):
        # Change the seed on each attempt so Ollama does not return
        # the same result repeatedly.
        attempt_seed = None if seed is None else seed + attempt - 1

        edited = ask_llm(
            system_prompt=system_prompt,
            user_prompt=build_editor_prompt(
                script=script,
                target_wpm=target_wpm,
                minimum_seconds=minimum_seconds,
                maximum_seconds=maximum_seconds,
            ),
            schema=VideoScript,
            temperature=temperature,
            seed=attempt_seed,
        )

        if len(edited.sections) != len(script.sections):
            print(
                f"Attempt {attempt}: expected {len(script.sections)} "
                f"sections, received {len(edited.sections)}. Retrying..."
            )
            continue

        # Keep the exact original topic object.
        edited.topic = script.topic

        edited = normalize_script(
            edited,
            words_per_minute=target_wpm,
        )

        if edited.word_count >= minimum_words:
            return edited

        print(
            f"Attempt {attempt}: generated {edited.word_count} words. "
            f"Minimum is {minimum_words}. Retrying..."
        )

    raise ValueError(
        f"Failed to generate a script with at least {minimum_words} words "
        f"after {max_attempts} attempts."
    )


def slugify(value: str) -> str:
    """Convert text into a filesystem-friendly slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "edited_script"


def build_edited_script_filename(script: VideoScript) -> str:
    """Create a stable output filename."""
    return f"{slugify(script.topic.title)}.json"


def save_script(
    script: VideoScript,
    output_path: Path,
) -> None:
    """Save a validated script as formatted UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(
            script.model_dump(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def summarize_changes(
    original: VideoScript,
    edited: VideoScript,
) -> dict[str, int]:
    """Return a compact before-and-after summary."""
    return {
        "original_word_count": original.word_count,
        "edited_word_count": edited.word_count,
        "word_count_change": edited.word_count - original.word_count,
        "original_seconds": original.estimated_total_seconds,
        "edited_seconds": edited.estimated_total_seconds,
        "seconds_change": (
            edited.estimated_total_seconds
            - original.estimated_total_seconds
        ),
    }