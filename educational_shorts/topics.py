from __future__ import annotations

import json
import random
import re
from pathlib import Path

from educational_shorts.client import ask_llm
from educational_shorts.schemas import KnowledgeNode, VideoTopicList


def collect_node_paths(
    tree: KnowledgeNode,
    min_depth: int = 1,
    max_depth: int | None = None,
) -> list[list[str]]:
    """Return category paths whose depths fall within the requested range."""
    if min_depth < 0:
        raise ValueError("min_depth must be zero or greater.")

    if max_depth is not None and max_depth < min_depth:
        raise ValueError("max_depth must be greater than or equal to min_depth.")

    paths: list[list[str]] = []

    def visit(node: KnowledgeNode, path: list[str], depth: int) -> None:
        current_path = [*path, node.name]

        if depth >= min_depth and (
            max_depth is None or depth <= max_depth
        ):
            paths.append(current_path)

        if max_depth is None or depth < max_depth:
            for child in node.children:
                visit(child, current_path, depth + 1)

    visit(tree, [], 0)
    return paths


def sample_node_path(
    tree: KnowledgeNode,
    min_depth: int = 2,
    max_depth: int = 3,
    seed: int | None = None,
) -> list[str]:
    """Select one category path from the requested tree-depth range."""
    candidates = collect_node_paths(
        tree=tree,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    if not candidates:
        raise ValueError(
            "No knowledge-tree nodes exist within the requested depth range. "
            "Expand the tree further or lower min_depth."
        )

    return random.Random(seed).choice(candidates)


def build_topic_prompt(
    category_path: list[str],
    count: int,
) -> str:
    """Build the user prompt for educational topic generation."""
    if count < 1:
        raise ValueError("count must be at least 1.")

    path_text = " > ".join(category_path)

    return f"""
Generate exactly {count} distinct educational short-video topics for this
knowledge-tree category:

{path_text}

Requirements:
- Keep every topic focused enough for one short educational video.
- Make the topics meaningfully different from one another.
- Use the exact category path provided above for every topic.
- Give each topic a clear one-sentence learning objective.
- Prefer concrete questions, mechanisms, comparisons, or surprising concepts.
- Avoid titles that are vague, excessively broad, or dependent on clickbait.
""".strip()


def generate_topics(
    category_path: list[str],
    system_prompt: str,
    count: int = 10,
    temperature: float = 0.7,
    seed: int | None = None,
) -> VideoTopicList:
    """Generate structured educational video topics for one category path."""
    return ask_llm(
        system_prompt=system_prompt,
        user_prompt=build_topic_prompt(
            category_path=category_path,
            count=count,
        ),
        schema=VideoTopicList,
        temperature=temperature,
        seed=seed,
    )


def slugify(value: str) -> str:
    """Convert text into a filesystem-friendly lowercase slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "topics"


def build_topics_filename(category_path: list[str]) -> str:
    """Create a filename that preserves the selected category path."""
    return "__".join(slugify(part) for part in category_path) + ".json"


def save_topics(
    topics: VideoTopicList,
    output_path: Path,
) -> None:
    """Save generated topics as formatted UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(
            topics.model_dump(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_topics(input_path: Path) -> VideoTopicList:
    """Load and validate a previously saved topic collection."""
    return VideoTopicList.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )