from __future__ import annotations

import json
import re
from pathlib import Path

from educational_shorts.client import ask_llm
from educational_shorts.schemas import (
    VideoOutline,
    VideoTopic,
    VideoTopicList,
)


def find_topic(
    topic_list: VideoTopicList,
    title: str | None = None,
    index: int = 0,
) -> VideoTopic:
    """Select a topic by exact title, or by zero-based index."""
    if not topic_list.topics:
        raise ValueError("The topic collection is empty.")

    if title is not None:
        for topic in topic_list.topics:
            if topic.title.casefold() == title.casefold():
                return topic

        available_titles = [topic.title for topic in topic_list.topics]
        raise ValueError(
            f"Topic titled {title!r} was not found. "
            f"Available topics: {available_titles}"
        )

    if index < 0 or index >= len(topic_list.topics):
        raise IndexError(
            f"Topic index {index} is outside the valid range "
            f"0 to {len(topic_list.topics) - 1}."
        )

    return topic_list.topics[index]


def find_topic_file(
    topics_directory: Path,
    filename: str | None = None,
) -> Path:
    """Resolve a topic JSON file explicitly or choose the most recent one."""
    if filename is not None:
        path = topics_directory / filename

        if not path.exists():
            raise FileNotFoundError(f"Topic file not found: {path}")

        return path

    candidates = sorted(
        topics_directory.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No topic JSON files were found in {topics_directory}."
        )

    return candidates[0]


def build_outline_prompt(
    topic: VideoTopic,
    target_seconds: int = 60,
    section_count: int = 4,
) -> str:
    """Build the user prompt for structured outline generation."""
    if target_seconds < 15:
        raise ValueError("target_seconds must be at least 15.")

    if section_count < 2:
        raise ValueError("section_count must be at least 2.")

    category_path = " > ".join(topic.category_path)

    return f"""
Create a structured outline for one educational short video.

Topic title:
{topic.title}

Knowledge-tree category:
{category_path}

Learning objective:
{topic.learning_objective}

Requirements:
- Target a total duration of approximately {target_seconds} seconds.
- Write exactly {section_count} ordered body sections.
- Keep the hook separate from the body sections.
- Give every body section a clear purpose, essential key points, a useful
  visual direction, and an estimated duration.
- Build the explanation logically for a viewer with no specialist background.
- Keep the scope narrow enough for one short video.
- Do not write the complete narration yet.
- Avoid unsupported statistics, exaggerated claims, and unnecessary jargon.
- The estimated body-section times should be realistic for the target length.
- End with one concise closing takeaway.
""".strip()


def generate_outline(
    topic: VideoTopic,
    system_prompt: str,
    target_seconds: int = 60,
    section_count: int = 4,
    temperature: float = 0.4,
    seed: int | None = None,
) -> VideoOutline:
    """Generate and validate a structured outline for one video topic."""
    outline = ask_llm(
        system_prompt=system_prompt,
        user_prompt=build_outline_prompt(
            topic=topic,
            target_seconds=target_seconds,
            section_count=section_count,
        ),
        schema=VideoOutline,
        temperature=temperature,
        seed=seed,
    )

    # Preserve the exact selected topic, even if the model paraphrases it.
    outline.topic = topic
    return outline


def slugify(value: str) -> str:
    """Convert text into a filesystem-friendly lowercase slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "outline"


def build_outline_filename(topic: VideoTopic) -> str:
    """Create a stable JSON filename from a topic title."""
    return f"{slugify(topic.title)}.json"


def save_outline(
    outline: VideoOutline,
    output_path: Path,
) -> None:
    """Save a validated outline as formatted UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(
            outline.model_dump(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_outline(input_path: Path) -> VideoOutline:
    """Load and validate a previously saved outline."""
    return VideoOutline.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )