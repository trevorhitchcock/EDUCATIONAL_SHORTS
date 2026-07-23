"""NOTEBOOK_10_CAPTION_GENERATION_FRESH_V1

Deterministic caption generation for Educational Shorts.

This module uses the exact TTS segment durations from Notebook 09 and the
corresponding checked narration text. It does not run speech recognition.

Outputs:
- word-level timing JSON
- caption cue JSON
- SRT
- WebVTT
- ASS with centered short-form styling
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from educational_shorts.tts import TTSManifest


class WordTiming(BaseModel):
    word: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    segment_index: int = Field(ge=0)
    segment_type: str


class CaptionCue(BaseModel):
    cue_index: int = Field(ge=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str
    word_count: int = Field(ge=1)
    segment_index: int = Field(ge=0)
    segment_type: str


class CaptionManifest(BaseModel):
    source_topic_title: str
    source_tts_manifest_filename: str
    source_audio_filename: str

    output_directory: str
    caption_json_filename: str
    word_timing_filename: str
    srt_filename: str
    vtt_filename: str
    ass_filename: str

    caption_style: Literal["phrase", "word", "karaoke"]
    max_words_per_cue: int
    max_characters_per_line: int
    max_lines: int
    minimum_cue_seconds: float
    maximum_cue_seconds: float

    cue_count: int
    word_count: int
    audio_duration_seconds: float
    final_caption_end_seconds: float

    generated_at_utc: str
    cues: list[CaptionCue]


def find_tts_manifest(
    audio_directory: Path,
    filename: str | None = None,
) -> Path:
    """Use an explicit TTS manifest or the newest manifest recursively."""
    if filename:
        path = audio_directory / filename

        if not path.exists():
            raise FileNotFoundError(f"TTS manifest not found: {path}")

        return path

    candidates = sorted(
        audio_directory.rglob("tts_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No tts_manifest.json files found under {audio_directory}."
        )

    return candidates[0]


def load_tts_manifest(path: Path) -> TTSManifest:
    return TTSManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _tokenize_spoken_words(text: str) -> list[str]:
    """Tokenize while preserving punctuation attached to spoken words."""
    return re.findall(
        r"\b[\w’'-]+(?:[.,!?;:])?",
        text,
        flags=re.UNICODE,
    )


def _word_weight(word: str) -> float:
    """Approximate speaking time using character and punctuation weight."""
    stripped = re.sub(r"[^\w’'-]", "", word)
    weight = max(len(stripped), 1) ** 0.72

    if word.endswith((",", ";", ":")):
        weight += 1.4
    elif word.endswith((".", "!", "?")):
        weight += 2.3

    return weight


def build_word_timings(
    tts_manifest: TTSManifest,
) -> list[WordTiming]:
    """Distribute each segment duration across its exact TTS words."""
    timings: list[WordTiming] = []
    cursor = 0.0
    pause_seconds = tts_manifest.segment_pause_ms / 1000

    for segment_position, segment in enumerate(tts_manifest.segments):
        words = _tokenize_spoken_words(segment.tts_text)

        if not words:
            cursor += segment.actual_seconds

            if segment_position < len(tts_manifest.segments) - 1:
                cursor += pause_seconds

            continue

        weights = [_word_weight(word) for word in words]
        total_weight = sum(weights)
        segment_start = cursor
        local_cursor = segment_start

        for index, (word, weight) in enumerate(
            zip(words, weights, strict=True)
        ):
            share = segment.actual_seconds * weight / total_weight
            word_end = (
                segment_start + segment.actual_seconds
                if index == len(words) - 1
                else local_cursor + share
            )

            timings.append(
                WordTiming(
                    word=word,
                    start_seconds=round(local_cursor, 3),
                    end_seconds=round(word_end, 3),
                    segment_index=segment.index,
                    segment_type=segment.segment_type,
                )
            )
            local_cursor = word_end

        cursor = segment_start + segment.actual_seconds

        if segment_position < len(tts_manifest.segments) - 1:
            cursor += pause_seconds

    return timings


def _visible_length(words: list[WordTiming]) -> int:
    return len(" ".join(word.word for word in words))


def _should_end_cue(
    current: list[WordTiming],
    next_word: WordTiming | None,
    *,
    max_words_per_cue: int,
    max_characters_per_line: int,
    max_lines: int,
    maximum_cue_seconds: float,
) -> bool:
    if not current:
        return False

    duration = current[-1].end_seconds - current[0].start_seconds

    if len(current) >= max_words_per_cue:
        return True

    if duration >= maximum_cue_seconds:
        return True

    if _visible_length(current) >= (
        max_characters_per_line * max_lines
    ):
        return True

    if current[-1].word.endswith((".", "!", "?")):
        return True

    if current[-1].word.endswith((",", ";", ":")) and len(current) >= 3:
        return True

    if next_word and next_word.segment_index != current[-1].segment_index:
        return True

    return False


def _split_long_text(
    words: list[str],
    max_characters_per_line: int,
    max_lines: int,
) -> str:
    """Greedily wrap caption text without exceeding the line count."""
    lines: list[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()

        if (
            current
            and len(candidate) > max_characters_per_line
            and len(lines) < max_lines - 1
        ):
            lines.append(current)
            current = word
        else:
            current = candidate

    if current:
        lines.append(current)

    return "\n".join(lines[:max_lines])


def build_caption_cues(
    word_timings: list[WordTiming],
    *,
    max_words_per_cue: int = 5,
    max_characters_per_line: int = 22,
    max_lines: int = 2,
    minimum_cue_seconds: float = 0.55,
    maximum_cue_seconds: float = 2.6,
    caption_style: Literal["phrase", "word", "karaoke"] = "phrase",
) -> list[CaptionCue]:
    """Group word timings into readable short-form caption cues."""
    if caption_style == "word":
        max_words_per_cue = 1

    cues: list[CaptionCue] = []
    current: list[WordTiming] = []

    for index, word in enumerate(word_timings):
        current.append(word)
        next_word = (
            word_timings[index + 1]
            if index + 1 < len(word_timings)
            else None
        )

        if not _should_end_cue(
            current,
            next_word,
            max_words_per_cue=max_words_per_cue,
            max_characters_per_line=max_characters_per_line,
            max_lines=max_lines,
            maximum_cue_seconds=maximum_cue_seconds,
        ):
            continue

        start = current[0].start_seconds
        end = current[-1].end_seconds

        if end - start < minimum_cue_seconds:
            end = start + minimum_cue_seconds

            if next_word is not None:
                end = min(end, next_word.start_seconds)

        text = _split_long_text(
            [item.word for item in current],
            max_characters_per_line=max_characters_per_line,
            max_lines=max_lines,
        )

        cues.append(
            CaptionCue(
                cue_index=len(cues) + 1,
                start_seconds=round(start, 3),
                end_seconds=round(max(end, start + 0.05), 3),
                text=text,
                word_count=len(current),
                segment_index=current[0].segment_index,
                segment_type=current[0].segment_type,
            )
        )
        current = []

    if current:
        start = current[0].start_seconds
        end = max(
            current[-1].end_seconds,
            start + minimum_cue_seconds,
        )
        text = _split_long_text(
            [item.word for item in current],
            max_characters_per_line=max_characters_per_line,
            max_lines=max_lines,
        )

        cues.append(
            CaptionCue(
                cue_index=len(cues) + 1,
                start_seconds=round(start, 3),
                end_seconds=round(end, 3),
                text=text,
                word_count=len(current),
                segment_index=current[0].segment_index,
                segment_type=current[0].segment_type,
            )
        )

    return _remove_overlaps(cues)


def _remove_overlaps(
    cues: list[CaptionCue],
    gap_seconds: float = 0.025,
) -> list[CaptionCue]:
    """Guarantee ordered, non-overlapping caption intervals."""
    for index in range(len(cues) - 1):
        current = cues[index]
        following = cues[index + 1]
        maximum_end = following.start_seconds - gap_seconds

        if current.end_seconds > maximum_end:
            current.end_seconds = round(
                max(current.start_seconds + 0.05, maximum_end),
                3,
            )

    return cues


def _srt_timestamp(seconds: float) -> str:
    milliseconds = int(round(max(seconds, 0) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _vtt_timestamp(seconds: float) -> str:
    return _srt_timestamp(seconds).replace(",", ".")


def _ass_timestamp(seconds: float) -> str:
    centiseconds = int(round(max(seconds, 0) * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centis = divmod(remainder, 100)
    return f"{hours:d}:{minutes:02}:{secs:02}.{centis:02}"


def render_srt(cues: list[CaptionCue]) -> str:
    blocks: list[str] = []

    for cue in cues:
        blocks.append(
            "\n".join(
                [
                    str(cue.cue_index),
                    (
                        f"{_srt_timestamp(cue.start_seconds)} --> "
                        f"{_srt_timestamp(cue.end_seconds)}"
                    ),
                    cue.text,
                ]
            )
        )

    return "\n\n".join(blocks) + "\n"


def render_vtt(cues: list[CaptionCue]) -> str:
    blocks = ["WEBVTT", ""]

    for cue in cues:
        blocks.extend(
            [
                (
                    f"{_vtt_timestamp(cue.start_seconds)} --> "
                    f"{_vtt_timestamp(cue.end_seconds)}"
                ),
                cue.text,
                "",
            ]
        )

    return "\n".join(blocks).rstrip() + "\n"


def render_ass(
    cues: list[CaptionCue],
    *,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    font_name: str = "Arial",
    font_size: int = 72,
    margin_v: int = 300,
) -> str:
    """Render burn-in-ready ASS captions for vertical video."""
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,5,1,2,70,70,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: list[str] = []

    for cue in cues:
        text = cue.text.replace("\n", r"\N")
        events.append(
            "Dialogue: 0,"
            f"{_ass_timestamp(cue.start_seconds)},"
            f"{_ass_timestamp(cue.end_seconds)},"
            f"Default,,0,0,0,,{text}"
        )

    return header + "\n".join(events) + "\n"


def generate_caption_assets(
    tts_manifest: TTSManifest,
    source_tts_manifest_path: Path,
    output_root: Path,
    *,
    caption_style: Literal["phrase", "word", "karaoke"] = "phrase",
    max_words_per_cue: int = 5,
    max_characters_per_line: int = 22,
    max_lines: int = 2,
    minimum_cue_seconds: float = 0.55,
    maximum_cue_seconds: float = 2.6,
    ass_font_name: str = "Arial",
    ass_font_size: int = 72,
    ass_margin_v: int = 300,
) -> CaptionManifest:
    """Generate all caption formats and a validated manifest."""
    video_slug = source_tts_manifest_path.parent.name
    output_directory = output_root / video_slug
    output_directory.mkdir(parents=True, exist_ok=True)

    word_timings = build_word_timings(tts_manifest)
    cues = build_caption_cues(
        word_timings,
        max_words_per_cue=max_words_per_cue,
        max_characters_per_line=max_characters_per_line,
        max_lines=max_lines,
        minimum_cue_seconds=minimum_cue_seconds,
        maximum_cue_seconds=maximum_cue_seconds,
        caption_style=caption_style,
    )

    word_timing_path = output_directory / "word_timings.json"
    caption_json_path = output_directory / "captions.json"
    srt_path = output_directory / "captions.srt"
    vtt_path = output_directory / "captions.vtt"
    ass_path = output_directory / "captions.ass"

    word_timing_path.write_text(
        json.dumps(
            [timing.model_dump() for timing in word_timings],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    caption_json_path.write_text(
        json.dumps(
            [cue.model_dump() for cue in cues],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    srt_path.write_text(render_srt(cues), encoding="utf-8")
    vtt_path.write_text(render_vtt(cues), encoding="utf-8")
    ass_path.write_text(
        render_ass(
            cues,
            font_name=ass_font_name,
            font_size=ass_font_size,
            margin_v=ass_margin_v,
        ),
        encoding="utf-8",
    )

    final_end = cues[-1].end_seconds if cues else 0.0

    manifest = CaptionManifest(
        source_topic_title=tts_manifest.source_topic_title,
        source_tts_manifest_filename=source_tts_manifest_path.name,
        source_audio_filename=tts_manifest.master_audio_filename,
        output_directory=str(output_directory),
        caption_json_filename=caption_json_path.name,
        word_timing_filename=word_timing_path.name,
        srt_filename=srt_path.name,
        vtt_filename=vtt_path.name,
        ass_filename=ass_path.name,
        caption_style=caption_style,
        max_words_per_cue=max_words_per_cue,
        max_characters_per_line=max_characters_per_line,
        max_lines=max_lines,
        minimum_cue_seconds=minimum_cue_seconds,
        maximum_cue_seconds=maximum_cue_seconds,
        cue_count=len(cues),
        word_count=len(word_timings),
        audio_duration_seconds=tts_manifest.actual_duration_seconds,
        final_caption_end_seconds=round(final_end, 3),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        cues=cues,
    )

    manifest_path = output_directory / "caption_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest.model_dump(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return manifest


def summarize_caption_manifest(
    manifest: CaptionManifest,
) -> dict[str, object]:
    average_words = (
        sum(cue.word_count for cue in manifest.cues)
        / manifest.cue_count
        if manifest.cue_count
        else 0.0
    )

    return {
        "topic": manifest.source_topic_title,
        "style": manifest.caption_style,
        "words": manifest.word_count,
        "cues": manifest.cue_count,
        "average_words_per_cue": round(average_words, 2),
        "audio_seconds": manifest.audio_duration_seconds,
        "caption_end_seconds": manifest.final_caption_end_seconds,
        "srt": manifest.srt_filename,
        "vtt": manifest.vtt_filename,
        "ass": manifest.ass_filename,
        "output_directory": manifest.output_directory,
    }
