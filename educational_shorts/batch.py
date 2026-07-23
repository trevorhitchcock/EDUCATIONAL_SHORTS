"""NOTEBOOK_12_BATCH_PIPELINE_FRESH_V1

End-to-end orchestration for the Educational Shorts project.

This module intentionally leaves the knowledge-tree expansion in Notebook 02.
It starts from an existing knowledge tree, prepares topic candidates, and then
runs selected topics through:

03 topic generation
04 outline generation
05 script generation
06 script editing
07 fact checking
08 metadata generation
09 Kokoro TTS
10 caption generation
11 video assembly
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from educational_shorts.captions import generate_caption_assets
from educational_shorts.editor import (
    build_edited_script_filename,
    edit_script,
    save_script as save_edited_script,
)
from educational_shorts.fact_checker import (
    build_checked_script_filename,
    build_report_filename,
    fact_check_script,
    save_report,
    save_script as save_checked_script,
)
from educational_shorts.metadata import (
    build_metadata_filename,
    generate_metadata,
    save_metadata,
)
from educational_shorts.outlines import (
    build_outline_filename,
    generate_outline,
    save_outline,
)
from educational_shorts.prompts import load_prompt
from educational_shorts.schemas import (
    KnowledgeNode,
    VideoTopic,
    VideoTopicList,
)
from educational_shorts.scripts import (
    build_script_filename,
    generate_script,
    save_script as save_generated_script,
)
from educational_shorts.topics import (
    build_topics_filename,
    generate_topics,
    sample_node_path,
    save_topics,
)
from educational_shorts.tts import (
    KokoroSynthesizer,
    generate_tts_assets,
)
from educational_shorts.video import (
    find_background_video,
    find_ffmpeg_executable,
    find_ffprobe_executable,
    render_video,
)


class BatchConfig(BaseModel):
    """Configuration shared across candidate generation and rendering."""

    root_category: str = "Science"
    tree_filename: str | None = None
    category_path_override: list[str] | None = None

    topic_candidate_count: int = Field(default=10, ge=1)
    min_category_depth: int = Field(default=2, ge=0)
    max_category_depth: int = Field(default=3, ge=0)
    category_selection_seed: int = 42
    topic_generation_seed: int = 42
    topic_generation_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
    )

    target_seconds: int = Field(default=60, ge=15)
    section_count: int = Field(default=4, ge=2)
    outline_temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    outline_seed: int = 42

    target_words_per_minute: int = Field(default=145, ge=80)
    script_temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    script_seed: int = 42

    editor_minimum_seconds: int = Field(default=40, ge=1)
    editor_maximum_seconds: int = Field(default=60, ge=1)
    editor_temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    editor_seed: int = 42

    fact_check_minimum_words: int = Field(default=105, ge=1)
    fact_check_maximum_words: int = Field(default=150, ge=1)
    fact_check_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
    )
    fact_check_seed: int = 42

    metadata_temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    metadata_seed: int = 42
    metadata_max_attempts: int = Field(default=4, ge=1)

    # When True, a topic marked for manual review is saved through Notebook 07,
    # then held before public metadata, audio, captions, and video are created.
    pause_on_manual_review: bool = True

    tts_language_code: str = "a"
    tts_voice: str = "am_michael"
    tts_speed: float = Field(default=1.0, gt=0)
    tts_chunk_pause_ms: int = Field(default=80, ge=0)
    tts_segment_pause_ms: int = Field(default=240, ge=0)
    tts_target_peak_dbfs: float = -1.0
    normalize_contextual_years: bool = True
    trim_tts_edge_silence: bool = True
    pronunciation_replacements: dict[str, str] = Field(
        default_factory=dict
    )

    caption_style: Literal["phrase", "word", "karaoke"] = "phrase"
    max_words_per_cue: int = Field(default=5, ge=1)
    max_characters_per_line: int = Field(default=22, ge=1)
    max_caption_lines: int = Field(default=2, ge=1)
    minimum_cue_seconds: float = Field(default=0.55, gt=0)
    maximum_cue_seconds: float = Field(default=2.6, gt=0)
    ass_font_name: str = "Arial"
    ass_font_size: int = Field(default=72, ge=1)
    ass_margin_v: int = Field(default=300, ge=0)

    gameplay_subdirectory: str = "subway_surfers"
    background_video_filename: str | None = None

    output_width: int = Field(default=1080, ge=2)
    output_height: int = Field(default=1920, ge=2)
    crop_anchor_y: Literal["top", "center"] = "top"
    video_random_seed: int = 42
    duration_extra_seconds: float = Field(default=0.25, ge=0)
    start_padding_seconds: float = Field(default=1.0, ge=0)
    end_padding_seconds: float = Field(default=1.0, ge=0)
    loop_background: bool = True
    video_crf: int = Field(default=20, ge=0, le=51)
    video_preset: str = "medium"
    audio_bitrate: str = "192k"

    skip_existing_final: bool = True
    continue_on_error: bool = False

    def model_post_init(self, __context: object) -> None:
        if self.max_category_depth < self.min_category_depth:
            raise ValueError(
                "max_category_depth must be at least min_category_depth."
            )

        if self.editor_maximum_seconds < self.editor_minimum_seconds:
            raise ValueError(
                "editor_maximum_seconds must be at least "
                "editor_minimum_seconds."
            )

        if self.fact_check_maximum_words < self.fact_check_minimum_words:
            raise ValueError(
                "fact_check_maximum_words must be at least "
                "fact_check_minimum_words."
            )

        if self.maximum_cue_seconds < self.minimum_cue_seconds:
            raise ValueError(
                "maximum_cue_seconds must be at least "
                "minimum_cue_seconds."
            )


class TopicBatchPlan(BaseModel):
    run_id: str
    category_path: list[str]
    topics_file: str
    created_at_utc: str
    topics: VideoTopicList


class PipelineItemResult(BaseModel):
    item_number: int = Field(ge=1)
    topic_index: int = Field(ge=0)
    topic_title: str

    status: Literal[
        "completed",
        "manual_review",
        "skipped_existing",
        "failed",
    ]
    final_stage: str
    message: str

    fact_check_verdict: str | None = None
    requires_manual_review: bool | None = None

    stages_completed: list[str] = Field(default_factory=list)
    paths: dict[str, str] = Field(default_factory=dict)

    started_at_utc: str
    finished_at_utc: str


class BatchRunManifest(BaseModel):
    marker: str = "NOTEBOOK_12_BATCH_PIPELINE_FRESH_V1"
    run_id: str
    category_path: list[str]
    selected_topic_indexes: list[int]
    selected_topic_titles: list[str]

    status: Literal[
        "completed",
        "completed_with_holds",
        "completed_with_errors",
        "aborted",
    ]

    output_directory: str
    manifest_filename: str

    started_at_utc: str
    finished_at_utc: str | None = None

    config: dict[str, object]
    items: list[PipelineItemResult] = Field(default_factory=list)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _slugify(value: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "batch"


def resolve_project_directories(project_root: Path) -> dict[str, Path]:
    data = project_root / "data"

    return {
        "data": data,
        "knowledge_tree": data / "knowledge_tree",
        "topics": data / "topics",
        "outlines": data / "outlines",
        "scripts": data / "scripts",
        "edited_scripts": data / "edited_scripts",
        "checked_scripts": data / "checked_scripts",
        "fact_checks": data / "fact_checks",
        "metadata": data / "metadata",
        "audio": data / "audio",
        "captions": data / "captions",
        "gameplay": data / "gameplay",
        "videos": data / "videos",
        "batch_runs": data / "batch_runs",
    }


def resolve_tree_path(
    project_root: Path,
    config: BatchConfig,
) -> Path:
    directories = resolve_project_directories(project_root)

    filename = (
        config.tree_filename
        or f"{_slugify(config.root_category)}.json"
    )
    return directories["knowledge_tree"] / filename


def load_knowledge_tree(path: Path) -> KnowledgeNode:
    if not path.exists():
        raise FileNotFoundError(
            "Knowledge tree not found. Run Notebook 02 first or set "
            f"tree_filename explicitly. Expected: {path}"
        )

    return KnowledgeNode.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def preflight_batch_pipeline(
    project_root: Path,
    config: BatchConfig,
) -> dict[str, object]:
    """Validate external tools and required project inputs."""
    tree_path = resolve_tree_path(project_root, config)
    tree = load_knowledge_tree(tree_path)

    directories = resolve_project_directories(project_root)
    gameplay_directory = (
        directories["gameplay"] / config.gameplay_subdirectory
    )

    background_video = find_background_video(
        gameplay_directory=gameplay_directory,
        filename=config.background_video_filename,
    )

    ffmpeg = find_ffmpeg_executable()
    ffprobe = find_ffprobe_executable()

    return {
        "project_root": str(project_root),
        "tree_path": str(tree_path),
        "tree_root": tree.name,
        "gameplay_directory": str(gameplay_directory),
        "background_video": str(background_video),
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
    }


def prepare_topic_batch(
    project_root: Path,
    config: BatchConfig,
) -> TopicBatchPlan:
    """Generate and save a candidate topic collection."""
    tree_path = resolve_tree_path(project_root, config)
    tree = load_knowledge_tree(tree_path)

    if config.category_path_override:
        category_path = list(config.category_path_override)
    else:
        category_path = sample_node_path(
            tree=tree,
            min_depth=config.min_category_depth,
            max_depth=config.max_category_depth,
            seed=config.category_selection_seed,
        )

    topic_prompt = load_prompt("topic_generation")

    topics = generate_topics(
        category_path=category_path,
        system_prompt=topic_prompt,
        count=config.topic_candidate_count,
        temperature=config.topic_generation_temperature,
        seed=config.topic_generation_seed,
    )

    run_id = _run_id()
    directories = resolve_project_directories(project_root)

    standard_filename = build_topics_filename(category_path)
    standard_stem = Path(standard_filename).stem
    topics_path = (
        directories["topics"]
        / f"{standard_stem}__{run_id}.json"
    )

    save_topics(
        topics=topics,
        output_path=topics_path,
    )

    return TopicBatchPlan(
        run_id=run_id,
        category_path=category_path,
        topics_file=str(topics_path),
        created_at_utc=_utc_now(),
        topics=topics,
    )


def _load_pipeline_prompts() -> dict[str, str]:
    return {
        "outline": load_prompt("outline_generation"),
        "script": load_prompt("script_generation"),
        "editor": load_prompt("script_editor"),
        "fact_checker": load_prompt("fact_checker"),
        "metadata": load_prompt("metadata_generation"),
    }


def _find_existing_render_for_topic(
    videos_directory: Path,
    topic_title: str,
) -> Path | None:
    if not videos_directory.exists():
        return None

    for manifest_path in videos_directory.rglob("render_manifest.json"):
        try:
            payload = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue

        if payload.get("source_topic_title") != topic_title:
            continue

        final_name = payload.get("output_video_filename", "final.mp4")
        final_path = manifest_path.parent / final_name

        if final_path.exists():
            return final_path

    return None


def _save_batch_manifest(
    manifest: BatchRunManifest,
    manifest_path: Path,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _validate_selected_indexes(
    topics: VideoTopicList,
    selected_topic_indexes: list[int],
) -> list[int]:
    if not selected_topic_indexes:
        raise ValueError(
            "selected_topic_indexes cannot be empty. Choose at least one "
            "candidate topic index."
        )

    unique_indexes: list[int] = []
    seen: set[int] = set()

    for index in selected_topic_indexes:
        if index in seen:
            continue

        if index < 0 or index >= len(topics.topics):
            raise IndexError(
                f"Topic index {index} is outside the valid range "
                f"0 to {len(topics.topics) - 1}."
            )

        seen.add(index)
        unique_indexes.append(index)

    return unique_indexes


def _run_one_topic(
    *,
    project_root: Path,
    config: BatchConfig,
    prompts: dict[str, str],
    background_video_path: Path,
    topic: VideoTopic,
    topic_index: int,
    item_number: int,
    synthesizer: KokoroSynthesizer | None,
) -> tuple[PipelineItemResult, KokoroSynthesizer | None]:
    directories = resolve_project_directories(project_root)
    started = _utc_now()
    stages_completed: list[str] = []
    paths: dict[str, str] = {}
    stage = "initialization"

    existing = _find_existing_render_for_topic(
        videos_directory=directories["videos"],
        topic_title=topic.title,
    )

    if existing is not None and config.skip_existing_final:
        return (
            PipelineItemResult(
                item_number=item_number,
                topic_index=topic_index,
                topic_title=topic.title,
                status="skipped_existing",
                final_stage="existing_video_check",
                message="A completed video already exists for this topic.",
                stages_completed=["existing_video_check"],
                paths={"final_video": str(existing)},
                started_at_utc=started,
                finished_at_utc=_utc_now(),
            ),
            synthesizer,
        )

    try:
        seed_offset = item_number - 1

        stage = "outline_generation"
        outline = generate_outline(
            topic=topic,
            system_prompt=prompts["outline"],
            target_seconds=config.target_seconds,
            section_count=config.section_count,
            temperature=config.outline_temperature,
            seed=config.outline_seed + seed_offset,
        )
        outline_path = (
            directories["outlines"]
            / build_outline_filename(topic)
        )
        save_outline(outline=outline, output_path=outline_path)
        paths["outline"] = str(outline_path)
        stages_completed.append(stage)

        stage = "script_generation"
        script = generate_script(
            outline=outline,
            system_prompt=prompts["script"],
            target_wpm=config.target_words_per_minute,
            temperature=config.script_temperature,
            seed=config.script_seed + seed_offset,
        )
        script_path = (
            directories["scripts"]
            / build_script_filename(outline)
        )
        save_generated_script(
            script=script,
            output_path=script_path,
        )
        paths["script"] = str(script_path)
        stages_completed.append(stage)

        stage = "script_editing"
        edited_script = edit_script(
            script=script,
            system_prompt=prompts["editor"],
            target_wpm=config.target_words_per_minute,
            minimum_seconds=config.editor_minimum_seconds,
            maximum_seconds=config.editor_maximum_seconds,
            temperature=config.editor_temperature,
            seed=config.editor_seed + seed_offset,
        )
        edited_script_path = (
            directories["edited_scripts"]
            / build_edited_script_filename(edited_script)
        )
        save_edited_script(
            script=edited_script,
            output_path=edited_script_path,
        )
        paths["edited_script"] = str(edited_script_path)
        stages_completed.append(stage)

        stage = "fact_checking"
        report = fact_check_script(
            script=edited_script,
            system_prompt=prompts["fact_checker"],
            target_wpm=config.target_words_per_minute,
            minimum_words=config.fact_check_minimum_words,
            maximum_words=config.fact_check_maximum_words,
            temperature=config.fact_check_temperature,
            seed=config.fact_check_seed + seed_offset,
        )
        corrected_script = report.corrected_script

        checked_script_path = (
            directories["checked_scripts"]
            / build_checked_script_filename(corrected_script)
        )
        report_path = (
            directories["fact_checks"]
            / build_report_filename(corrected_script)
        )

        save_checked_script(
            script=corrected_script,
            output_path=checked_script_path,
        )
        save_report(
            report=report,
            output_path=report_path,
        )
        paths["checked_script"] = str(checked_script_path)
        paths["fact_check_report"] = str(report_path)
        stages_completed.append(stage)

        if (
            report.requires_manual_review
            and config.pause_on_manual_review
        ):
            return (
                PipelineItemResult(
                    item_number=item_number,
                    topic_index=topic_index,
                    topic_title=topic.title,
                    status="manual_review",
                    final_stage=stage,
                    message=(
                        "Fact checking completed, but the topic was held "
                        "before publication assets because manual review is "
                        "enabled."
                    ),
                    fact_check_verdict=report.verdict,
                    requires_manual_review=True,
                    stages_completed=stages_completed,
                    paths=paths,
                    started_at_utc=started,
                    finished_at_utc=_utc_now(),
                ),
                synthesizer,
            )

        stage = "metadata_generation"
        metadata = generate_metadata(
            script=corrected_script,
            report=report,
            system_prompt=prompts["metadata"],
            source_script_filename=checked_script_path.name,
            fact_check_report_filename=report_path.name,
            temperature=config.metadata_temperature,
            seed=config.metadata_seed + seed_offset,
            max_attempts=config.metadata_max_attempts,
        )
        metadata_path = (
            directories["metadata"]
            / build_metadata_filename(metadata)
        )
        save_metadata(
            metadata=metadata,
            output_path=metadata_path,
        )
        paths["metadata"] = str(metadata_path)
        stages_completed.append(stage)

        stage = "tts_generation"
        if synthesizer is None:
            synthesizer = KokoroSynthesizer(
                language_code=config.tts_language_code,
                voice=config.tts_voice,
                speed=config.tts_speed,
                chunk_pause_ms=config.tts_chunk_pause_ms,
            )

        tts_result = generate_tts_assets(
            script=corrected_script,
            metadata=metadata,
            metadata_filename=metadata_path.name,
            output_root=directories["audio"],
            synthesizer=synthesizer,
            segment_pause_ms=config.tts_segment_pause_ms,
            pronunciation_replacements=(
                config.pronunciation_replacements
            ),
            normalize_contextual_years=(
                config.normalize_contextual_years
            ),
            trim_edge_silence=config.trim_tts_edge_silence,
            target_peak_dbfs=config.tts_target_peak_dbfs,
        )
        paths["tts_manifest"] = str(tts_result.manifest_path)
        paths["narration"] = str(tts_result.master_audio_path)
        paths["transcript"] = str(tts_result.transcript_path)
        stages_completed.append(stage)

        stage = "caption_generation"
        caption_manifest = generate_caption_assets(
            tts_manifest=tts_result.manifest,
            source_tts_manifest_path=tts_result.manifest_path,
            output_root=directories["captions"],
            caption_style=config.caption_style,
            max_words_per_cue=config.max_words_per_cue,
            max_characters_per_line=(
                config.max_characters_per_line
            ),
            max_lines=config.max_caption_lines,
            minimum_cue_seconds=config.minimum_cue_seconds,
            maximum_cue_seconds=config.maximum_cue_seconds,
            ass_font_name=config.ass_font_name,
            ass_font_size=config.ass_font_size,
            ass_margin_v=config.ass_margin_v,
        )
        caption_directory = Path(caption_manifest.output_directory)
        caption_manifest_path = (
            caption_directory / "caption_manifest.json"
        )
        captions_ass_path = (
            caption_directory / caption_manifest.ass_filename
        )
        paths["caption_manifest"] = str(caption_manifest_path)
        paths["captions_ass"] = str(captions_ass_path)
        stages_completed.append(stage)

        stage = "video_assembly"
        render_manifest = render_video(
            project_root=project_root,
            metadata_path=metadata_path,
            background_video_path=background_video_path,
            output_root=directories["videos"],
            output_width=config.output_width,
            output_height=config.output_height,
            crop_anchor_y=config.crop_anchor_y,
            random_seed=config.video_random_seed + seed_offset,
            duration_extra_seconds=config.duration_extra_seconds,
            start_padding_seconds=config.start_padding_seconds,
            end_padding_seconds=config.end_padding_seconds,
            loop_background=config.loop_background,
            crf=config.video_crf,
            preset=config.video_preset,
            audio_bitrate=config.audio_bitrate,
        )
        render_directory = Path(render_manifest.output_directory)
        final_video_path = (
            render_directory
            / render_manifest.output_video_filename
        )
        render_manifest_path = (
            render_directory / "render_manifest.json"
        )
        paths["render_manifest"] = str(render_manifest_path)
        paths["final_video"] = str(final_video_path)
        stages_completed.append(stage)

        return (
            PipelineItemResult(
                item_number=item_number,
                topic_index=topic_index,
                topic_title=topic.title,
                status="completed",
                final_stage=stage,
                message="Finished the complete educational-short pipeline.",
                fact_check_verdict=report.verdict,
                requires_manual_review=(
                    report.requires_manual_review
                ),
                stages_completed=stages_completed,
                paths=paths,
                started_at_utc=started,
                finished_at_utc=_utc_now(),
            ),
            synthesizer,
        )

    except Exception as error:
        return (
            PipelineItemResult(
                item_number=item_number,
                topic_index=topic_index,
                topic_title=topic.title,
                status="failed",
                final_stage=stage,
                message=f"{type(error).__name__}: {error}",
                stages_completed=stages_completed,
                paths=paths,
                started_at_utc=started,
                finished_at_utc=_utc_now(),
            ),
            synthesizer,
        )


def run_batch_pipeline(
    *,
    project_root: Path,
    plan: TopicBatchPlan,
    selected_topic_indexes: list[int],
    config: BatchConfig,
) -> BatchRunManifest:
    """Run selected candidate topics through the complete pipeline."""
    indexes = _validate_selected_indexes(
        topics=plan.topics,
        selected_topic_indexes=selected_topic_indexes,
    )
    selected_topics = [
        plan.topics.topics[index]
        for index in indexes
    ]

    directories = resolve_project_directories(project_root)
    gameplay_directory = (
        directories["gameplay"] / config.gameplay_subdirectory
    )
    background_video_path = find_background_video(
        gameplay_directory=gameplay_directory,
        filename=config.background_video_filename,
    )

    # Fail early before any expensive LLM or TTS calls.
    find_ffmpeg_executable()
    find_ffprobe_executable()

    prompts = _load_pipeline_prompts()

    run_id = _run_id()
    output_directory = directories["batch_runs"] / run_id
    manifest_path = output_directory / "batch_manifest.json"

    manifest = BatchRunManifest(
        run_id=run_id,
        category_path=plan.category_path,
        selected_topic_indexes=indexes,
        selected_topic_titles=[
            topic.title
            for topic in selected_topics
        ],
        status="completed",
        output_directory=str(output_directory),
        manifest_filename=manifest_path.name,
        started_at_utc=_utc_now(),
        config=config.model_dump(),
    )
    _save_batch_manifest(manifest, manifest_path)

    synthesizer: KokoroSynthesizer | None = None
    aborted = False

    for item_number, (topic_index, topic) in enumerate(
        zip(indexes, selected_topics, strict=True),
        start=1,
    ):
        print()
        print("=" * 72)
        print(
            f"ITEM {item_number}/{len(selected_topics)}: "
            f"{topic.title}"
        )
        print("=" * 72)

        result, synthesizer = _run_one_topic(
            project_root=project_root,
            config=config,
            prompts=prompts,
            background_video_path=background_video_path,
            topic=topic,
            topic_index=topic_index,
            item_number=item_number,
            synthesizer=synthesizer,
        )
        manifest.items.append(result)
        _save_batch_manifest(manifest, manifest_path)

        print(f"Status: {result.status}")
        print(f"Final stage: {result.final_stage}")
        print(result.message)

        if (
            result.status == "failed"
            and not config.continue_on_error
        ):
            aborted = True
            break

    if aborted:
        manifest.status = "aborted"
    elif any(item.status == "failed" for item in manifest.items):
        manifest.status = "completed_with_errors"
    elif any(
        item.status == "manual_review"
        for item in manifest.items
    ):
        manifest.status = "completed_with_holds"
    else:
        manifest.status = "completed"

    manifest.finished_at_utc = _utc_now()
    _save_batch_manifest(manifest, manifest_path)
    return manifest


def summarize_topic_plan(
    plan: TopicBatchPlan,
) -> dict[str, object]:
    return {
        "candidate_run_id": plan.run_id,
        "category_path": " > ".join(plan.category_path),
        "candidate_count": len(plan.topics.topics),
        "topics_file": plan.topics_file,
    }


def summarize_batch_manifest(
    manifest: BatchRunManifest,
) -> dict[str, object]:
    counts = {
        status: sum(
            item.status == status
            for item in manifest.items
        )
        for status in [
            "completed",
            "manual_review",
            "skipped_existing",
            "failed",
        ]
    }

    return {
        "run_id": manifest.run_id,
        "status": manifest.status,
        "selected_topics": len(manifest.selected_topic_indexes),
        **counts,
        "output_directory": manifest.output_directory,
        "manifest": str(
            Path(manifest.output_directory)
            / manifest.manifest_filename
        ),
    }
