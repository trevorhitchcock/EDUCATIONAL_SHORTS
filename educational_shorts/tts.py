"""NOTEBOOK_09_TTS_GENERATION_KOKORO_V1

Local text-to-speech generation for Educational Shorts using Kokoro.

The module:
- loads publication metadata and its checked script
- synthesizes every narration segment separately
- applies one consistent peak-normalization gain
- writes segment WAV files and a combined narration WAV
- records timing and traceability in a JSON manifest
"""

from __future__ import annotations

import html
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field

from educational_shorts.metadata import VideoMetadata
from educational_shorts.schemas import ScriptSegment, VideoScript


DEFAULT_SAMPLE_RATE = 24_000


class TTSSegmentRecord(BaseModel):
    index: int = Field(ge=0)
    segment_type: str
    source_text: str
    tts_text: str
    audio_filename: str
    estimated_seconds: int = Field(ge=1)
    actual_seconds: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    sample_count: int = Field(gt=0)


class TTSManifest(BaseModel):
    model_name: str
    voice: str
    language_code: str
    speed: float

    source_topic_title: str
    source_metadata_filename: str
    source_script_filename: str

    output_directory: str
    master_audio_filename: str
    transcript_filename: str

    sample_rate: int
    segment_pause_ms: int
    chunk_pause_ms: int
    target_peak_dbfs: float
    actual_peak_dbfs: float

    script_estimated_seconds: int
    actual_duration_seconds: float
    duration_ratio: float

    generated_at_utc: str
    segments: list[TTSSegmentRecord]


@dataclass
class TTSGenerationResult:
    manifest: TTSManifest
    manifest_path: Path
    master_audio_path: Path
    transcript_path: Path
    segment_paths: list[Path]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "audio"


def find_metadata_file(
    metadata_directory: Path,
    filename: str | None = None,
) -> Path:
    """Use an explicit metadata file or the newest metadata JSON."""
    if filename:
        path = metadata_directory / filename

        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")

        return path

    candidates = sorted(
        metadata_directory.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No metadata JSON files found in {metadata_directory}."
        )

    return candidates[0]


def load_metadata(path: Path) -> VideoMetadata:
    return VideoMetadata.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def resolve_checked_script_path(
    metadata: VideoMetadata,
    checked_scripts_directory: Path,
) -> Path:
    path = checked_scripts_directory / metadata.source_script_filename

    if not path.exists():
        raise FileNotFoundError(
            "The checked script referenced by the metadata was not found: "
            f"{path}"
        )

    return path


def load_script(path: Path) -> VideoScript:
    return VideoScript.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def validate_script_metadata_pair(
    script: VideoScript,
    metadata: VideoMetadata,
) -> None:
    """Prevent synthesizing a script that does not match the selected metadata."""
    if script.topic.title != metadata.source_topic_title:
        raise ValueError(
            "Metadata/script topic mismatch: "
            f"{metadata.source_topic_title!r} != {script.topic.title!r}"
        )

    if script.word_count != metadata.script_word_count:
        raise ValueError(
            "Metadata/script word-count mismatch: "
            f"{metadata.script_word_count} != {script.word_count}. "
            "Regenerate Notebook 08 after changing the checked script."
        )


_SMALL_NUMBERS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}

_TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def _under_100_words(number: int) -> str:
    if number < 20:
        return _SMALL_NUMBERS[number]

    tens = number // 10 * 10
    remainder = number % 10

    if remainder == 0:
        return _TENS[tens]

    return f"{_TENS[tens]}-{_SMALL_NUMBERS[remainder]}"


def _year_to_words(year: int) -> str:
    """Render common historical/current years naturally for narration."""
    if 1500 <= year <= 1999:
        first = year // 100
        last = year % 100

        if last == 0:
            return f"{_under_100_words(first)} hundred"

        if last < 10:
            return (
                f"{_under_100_words(first)} oh "
                f"{_under_100_words(last)}"
            )

        return (
            f"{_under_100_words(first)} "
            f"{_under_100_words(last)}"
        )

    if 2000 <= year <= 2009:
        remainder = year - 2000

        if remainder == 0:
            return "two thousand"

        return f"two thousand {_under_100_words(remainder)}"

    if 2010 <= year <= 2099:
        return (
            f"twenty {_under_100_words(year - 2000)}"
        )

    return str(year)


_DATE_YEAR_PATTERN = re.compile(
    r"(?P<prefix>\b(?:"
    r"January|February|March|April|May|June|July|August|"
    r"September|October|November|December"
    r")\s+\d{1,2},\s+)"
    r"(?P<year>1[5-9]\d{2}|20\d{2})\b",
    flags=re.IGNORECASE,
)

_CONTEXT_YEAR_PATTERN = re.compile(
    r"(?P<prefix>\b(?:in|by|since|during|from|until|year)\s+)"
    r"(?P<year>1[5-9]\d{2}|20\d{2})\b",
    flags=re.IGNORECASE,
)


def _replace_contextual_years(text: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        return (
            f"{match.group('prefix')}"
            f"{_year_to_words(int(match.group('year')))}"
        )

    text = _DATE_YEAR_PATTERN.sub(replacement, text)
    text = _CONTEXT_YEAR_PATTERN.sub(replacement, text)
    return text


def prepare_tts_text(
    text: str,
    pronunciation_replacements: dict[str, str] | None = None,
    normalize_contextual_years: bool = True,
) -> str:
    """Create a spoken copy without changing the stored checked script."""
    prepared = html.unescape(text)
    prepared = prepared.replace("—", ", ")
    prepared = prepared.replace("–", "-")
    prepared = prepared.replace("“", '"').replace("”", '"')
    prepared = prepared.replace("’", "'")

    if normalize_contextual_years:
        prepared = _replace_contextual_years(prepared)

    for original, replacement in (
        pronunciation_replacements or {}
    ).items():
        prepared = prepared.replace(original, replacement)

    prepared = re.sub(r"\s+", " ", prepared).strip()
    return prepared


def iter_script_segments(
    script: VideoScript,
) -> list[tuple[str, ScriptSegment]]:
    segments: list[tuple[str, ScriptSegment]] = [
        ("hook", script.hook)
    ]

    for section in script.sections:
        segments.append((section.segment_type, section))

    segments.append(("closing", script.closing))
    return segments


def _silence(sample_rate: int, milliseconds: int) -> np.ndarray:
    samples = int(round(sample_rate * milliseconds / 1000))
    return np.zeros(max(samples, 0), dtype=np.float32)


def _trim_edge_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_dbfs: float = -45.0,
    keep_ms: int = 35,
) -> np.ndarray:
    """Trim only leading/trailing near-silence, preserving a short edge pad."""
    if audio.size == 0:
        return audio

    threshold = 10 ** (threshold_dbfs / 20)
    active = np.flatnonzero(np.abs(audio) >= threshold)

    if active.size == 0:
        return audio

    keep_samples = int(round(sample_rate * keep_ms / 1000))
    start = max(int(active[0]) - keep_samples, 0)
    end = min(int(active[-1]) + keep_samples + 1, audio.size)

    return audio[start:end]


def _peak_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return float("-inf")

    peak = float(np.max(np.abs(audio)))

    if peak <= 0:
        return float("-inf")

    return 20 * math.log10(peak)


def _normalization_gain(
    audio_arrays: list[np.ndarray],
    target_peak_dbfs: float,
) -> float:
    peak = max(
        (
            float(np.max(np.abs(audio)))
            for audio in audio_arrays
            if audio.size
        ),
        default=0.0,
    )

    if peak <= 0:
        return 1.0

    target_amplitude = 10 ** (target_peak_dbfs / 20)
    return target_amplitude / peak


class KokoroSynthesizer:
    """Thin wrapper that keeps one Kokoro pipeline loaded."""

    model_name = "hexgrad/Kokoro-82M"
    sample_rate = DEFAULT_SAMPLE_RATE

    def __init__(
        self,
        language_code: str = "a",
        voice: str = "am_michael",
        speed: float = 1.0,
        chunk_pause_ms: int = 80,
    ) -> None:
        if speed <= 0:
            raise ValueError("TTS speed must be greater than zero.")

        try:
            from kokoro import KPipeline
        except ImportError as error:
            raise ImportError(
                "Kokoro is not installed. Run: "
                'pip install "kokoro>=0.9.4" soundfile'
            ) from error

        self.language_code = language_code
        self.voice = voice
        self.speed = speed
        self.chunk_pause_ms = chunk_pause_ms
        self.pipeline = KPipeline(lang_code=language_code)

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize one script segment and concatenate Kokoro chunks."""
        arrays: list[np.ndarray] = []

        generator = self.pipeline(
            text,
            voice=self.voice,
            speed=self.speed,
        )

        for _, _, audio in generator:
            array = np.asarray(audio, dtype=np.float32).reshape(-1)

            if array.size:
                arrays.append(array)

        if not arrays:
            raise RuntimeError(
                f"Kokoro returned no audio for text: {text!r}"
            )

        if len(arrays) == 1:
            return arrays[0]

        pause = _silence(
            self.sample_rate,
            self.chunk_pause_ms,
        )

        parts: list[np.ndarray] = []

        for index, array in enumerate(arrays):
            if index:
                parts.append(pause)

            parts.append(array)

        return np.concatenate(parts).astype(np.float32)


def generate_tts_assets(
    script: VideoScript,
    metadata: VideoMetadata,
    metadata_filename: str,
    output_root: Path,
    synthesizer: KokoroSynthesizer,
    segment_pause_ms: int = 240,
    pronunciation_replacements: dict[str, str] | None = None,
    normalize_contextual_years: bool = True,
    trim_edge_silence: bool = True,
    target_peak_dbfs: float = -1.0,
) -> TTSGenerationResult:
    """Generate segment audio, master audio, transcript, and manifest."""
    validate_script_metadata_pair(script, metadata)

    output_directory = output_root / metadata.filename_slug
    output_directory.mkdir(parents=True, exist_ok=True)

    raw_records: list[dict[str, Any]] = []
    raw_audio: list[np.ndarray] = []

    script_segments = iter_script_segments(script)

    for index, (segment_type, segment) in enumerate(script_segments):
        tts_text = prepare_tts_text(
            segment.narration,
            pronunciation_replacements=pronunciation_replacements,
            normalize_contextual_years=normalize_contextual_years,
        )

        print(
            f"Synthesizing {index + 1}/{len(script_segments)}: "
            f"{segment_type}"
        )

        audio = synthesizer.synthesize(tts_text)

        if trim_edge_silence:
            audio = _trim_edge_silence(
                audio,
                sample_rate=synthesizer.sample_rate,
            )

        if audio.size == 0:
            raise RuntimeError(
                f"Audio became empty after processing: {segment_type}"
            )

        filename = (
            f"{index:02d}_{slugify(segment_type)}.wav"
        )

        raw_records.append(
            {
                "index": index,
                "segment_type": segment_type,
                "source_text": segment.narration,
                "tts_text": tts_text,
                "audio_filename": filename,
                "estimated_seconds": segment.estimated_seconds,
            }
        )
        raw_audio.append(audio)

    gain = _normalization_gain(
        raw_audio,
        target_peak_dbfs=target_peak_dbfs,
    )
    normalized_audio = [
        np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
        for audio in raw_audio
    ]

    segment_paths: list[Path] = []
    segment_records: list[TTSSegmentRecord] = []

    for record, audio in zip(
        raw_records,
        normalized_audio,
        strict=True,
    ):
        path = output_directory / record["audio_filename"]

        sf.write(
            path,
            audio,
            synthesizer.sample_rate,
            subtype="PCM_16",
        )

        segment_paths.append(path)
        segment_records.append(
            TTSSegmentRecord(
                **record,
                actual_seconds=round(
                    audio.size / synthesizer.sample_rate,
                    3,
                ),
                sample_rate=synthesizer.sample_rate,
                sample_count=int(audio.size),
            )
        )

    inter_segment_silence = _silence(
        synthesizer.sample_rate,
        segment_pause_ms,
    )

    master_parts: list[np.ndarray] = []

    for index, audio in enumerate(normalized_audio):
        if index:
            master_parts.append(inter_segment_silence)

        master_parts.append(audio)

    master_audio = np.concatenate(master_parts).astype(
        np.float32
    )

    master_audio_path = output_directory / "narration.wav"
    sf.write(
        master_audio_path,
        master_audio,
        synthesizer.sample_rate,
        subtype="PCM_16",
    )

    transcript_path = output_directory / "narration.txt"
    transcript_lines = [
        f"TITLE: {script.topic.title}",
        "",
    ]

    for record in segment_records:
        transcript_lines.extend(
            [
                record.segment_type.upper(),
                record.source_text,
                "",
            ]
        )

    transcript_path.write_text(
        "\n".join(transcript_lines).strip() + "\n",
        encoding="utf-8",
    )

    actual_duration = (
        master_audio.size / synthesizer.sample_rate
    )
    expected_duration = script.estimated_total_seconds
    duration_ratio = (
        actual_duration / expected_duration
        if expected_duration > 0
        else 0.0
    )

    manifest = TTSManifest(
        model_name=synthesizer.model_name,
        voice=synthesizer.voice,
        language_code=synthesizer.language_code,
        speed=synthesizer.speed,
        source_topic_title=script.topic.title,
        source_metadata_filename=metadata_filename,
        source_script_filename=metadata.source_script_filename,
        output_directory=str(output_directory),
        master_audio_filename=master_audio_path.name,
        transcript_filename=transcript_path.name,
        sample_rate=synthesizer.sample_rate,
        segment_pause_ms=segment_pause_ms,
        chunk_pause_ms=synthesizer.chunk_pause_ms,
        target_peak_dbfs=target_peak_dbfs,
        actual_peak_dbfs=round(_peak_dbfs(master_audio), 3),
        script_estimated_seconds=expected_duration,
        actual_duration_seconds=round(actual_duration, 3),
        duration_ratio=round(duration_ratio, 3),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        segments=segment_records,
    )

    manifest_path = output_directory / "tts_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest.model_dump(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return TTSGenerationResult(
        manifest=manifest,
        manifest_path=manifest_path,
        master_audio_path=master_audio_path,
        transcript_path=transcript_path,
        segment_paths=segment_paths,
    )


def summarize_tts_result(
    result: TTSGenerationResult,
) -> dict[str, object]:
    manifest = result.manifest

    return {
        "topic": manifest.source_topic_title,
        "model": manifest.model_name,
        "voice": manifest.voice,
        "speed": manifest.speed,
        "segments": len(manifest.segments),
        "sample_rate": manifest.sample_rate,
        "estimated_seconds": manifest.script_estimated_seconds,
        "actual_seconds": manifest.actual_duration_seconds,
        "duration_ratio": manifest.duration_ratio,
        "peak_dbfs": manifest.actual_peak_dbfs,
        "master_audio": str(result.master_audio_path),
        "manifest": str(result.manifest_path),
    }
