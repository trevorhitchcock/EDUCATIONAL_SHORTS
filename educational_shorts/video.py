"""NOTEBOOK_11_VIDEO_ASSEMBLY_FRESH_V1

Assemble a finished vertical short from:
- gameplay/background footage
- narration.wav from Notebook 09
- captions.ass from Notebook 10
- metadata from Notebook 08

The default crop strategy is designed for iPhone portrait screen recordings:
scale to fill 1080x1920 and crop from the top-left/center so extra height is
removed from the bottom. This removes the bottom of tall iPhone recordings
while preserving the gameplay area near the top.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from educational_shorts.captions import CaptionManifest
from educational_shorts.metadata import VideoMetadata
from educational_shorts.tts import TTSManifest


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".webm",
    ".mkv",
}


class VideoProbe(BaseModel):
    path: str
    duration_seconds: float
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = None


class RenderManifest(BaseModel):
    source_topic_title: str
    video_slug: str

    background_video: str
    background_probe: VideoProbe
    selected_background_start_seconds: float
    required_duration_seconds: float

    narration_audio: str
    captions_ass: str
    metadata_file: str
    tts_manifest_file: str
    caption_manifest_file: str

    output_directory: str
    output_video_filename: str

    output_width: int
    output_height: int
    crop_anchor_y: Literal["top", "center"]
    loop_background: bool

    ffmpeg_command: list[str]
    generated_at_utc: str


def find_ffmpeg_executable() -> str:
    """Return ffmpeg executable name/path if available."""
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        raise FileNotFoundError(
            "ffmpeg was not found on PATH. Install FFmpeg and restart "
            "PowerShell/Jupyter."
        )

    return ffmpeg_path


def find_ffprobe_executable() -> str:
    """Return ffprobe executable name/path if available."""
    ffprobe_path = shutil.which("ffprobe")

    if ffprobe_path is None:
        raise FileNotFoundError(
            "ffprobe was not found on PATH. Install FFmpeg and restart "
            "PowerShell/Jupyter."
        )

    return ffprobe_path


def _run_json_command(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )

    return json.loads(completed.stdout)


def probe_video(path: Path) -> VideoProbe:
    """Read duration, dimensions, and frame rate from a video file."""
    ffprobe = find_ffprobe_executable()

    payload = _run_json_command(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )

    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}

    duration_raw = stream.get("duration") or fmt.get("duration") or 0
    duration = float(duration_raw)

    frame_rate = None
    frame_rate_raw = stream.get("r_frame_rate")

    if frame_rate_raw and "/" in frame_rate_raw:
        numerator, denominator = frame_rate_raw.split("/", 1)

        if float(denominator) != 0:
            frame_rate = float(numerator) / float(denominator)

    return VideoProbe(
        path=str(path),
        duration_seconds=round(duration, 3),
        width=stream.get("width"),
        height=stream.get("height"),
        frame_rate=round(frame_rate, 3) if frame_rate else None,
    )


def find_background_video(
    gameplay_directory: Path,
    filename: str | None = None,
) -> Path:
    """Use an explicit gameplay video or the newest file recursively."""
    if filename:
        path = gameplay_directory / filename

        if not path.exists():
            raise FileNotFoundError(f"Gameplay file not found: {path}")

        return path

    candidates = sorted(
        [
            path
            for path in gameplay_directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No gameplay videos were found under {gameplay_directory}."
        )

    return candidates[0]


def load_metadata(path: Path) -> VideoMetadata:
    return VideoMetadata.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def load_tts_manifest(path: Path) -> TTSManifest:
    return TTSManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def load_caption_manifest(path: Path) -> CaptionManifest:
    return CaptionManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )


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


def resolve_pipeline_paths(
    project_root: Path,
    metadata_path: Path,
) -> dict[str, Path]:
    """Resolve matching TTS/caption assets from the metadata slug."""
    metadata = load_metadata(metadata_path)
    slug = metadata.filename_slug

    audio_dir = project_root / "data" / "audio" / slug
    captions_dir = project_root / "data" / "captions" / slug

    paths = {
        "metadata": metadata_path,
        "audio_dir": audio_dir,
        "captions_dir": captions_dir,
        "tts_manifest": audio_dir / "tts_manifest.json",
        "narration": audio_dir / "narration.wav",
        "caption_manifest": captions_dir / "caption_manifest.json",
        "captions_ass": captions_dir / "captions.ass",
    }

    missing = [
        str(path)
        for path in paths.values()
        if path.suffix and not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required pipeline asset(s):\n"
            + "\n".join(missing)
        )

    return paths


def choose_background_start(
    background_duration: float,
    required_duration: float,
    start_padding_seconds: float = 1.0,
    end_padding_seconds: float = 1.0,
    seed: int | None = None,
) -> float:
    """Choose a random start if the clip is longer than the render."""
    usable = background_duration - required_duration - end_padding_seconds

    if usable <= start_padding_seconds:
        return 0.0

    rng = random.Random(seed)
    return round(
        rng.uniform(start_padding_seconds, usable),
        3,
    )


def _crop_y_expression(crop_anchor_y: Literal["top", "center"]) -> str:
    if crop_anchor_y == "top":
        return "0"

    if crop_anchor_y == "center":
        return "(ih-1920)/2"

    raise ValueError(f"Unsupported crop_anchor_y: {crop_anchor_y}")


def build_video_filter(
    captions_filename: str,
    output_width: int = 1080,
    output_height: int = 1920,
    crop_anchor_y: Literal["top", "center"] = "top",
) -> str:
    """Build the FFmpeg video filter graph.

    The ASS file is referenced by filename only. The subprocess cwd is set to
    the captions directory, which avoids Windows path escaping problems.
    """
    crop_y = _crop_y_expression(crop_anchor_y)

    return (
        f"scale={output_width}:{output_height}:"
        "force_original_aspect_ratio=increase,"
        f"crop={output_width}:{output_height}:"
        f"(iw-{output_width})/2:{crop_y},"
        "setsar=1,"
        f"ass={captions_filename}"
    )


def render_video(
    *,
    project_root: Path,
    metadata_path: Path,
    background_video_path: Path,
    output_root: Path,
    output_width: int = 1080,
    output_height: int = 1920,
    crop_anchor_y: Literal["top", "center"] = "top",
    random_seed: int | None = 42,
    duration_extra_seconds: float = 0.25,
    start_padding_seconds: float = 1.0,
    end_padding_seconds: float = 1.0,
    loop_background: bool = True,
    crf: int = 20,
    preset: str = "medium",
    audio_bitrate: str = "192k",
) -> RenderManifest:
    """Render a finished vertical video with narration and burned captions."""
    ffmpeg = find_ffmpeg_executable()

    paths = resolve_pipeline_paths(
        project_root=project_root,
        metadata_path=metadata_path,
    )

    metadata = load_metadata(paths["metadata"])
    tts_manifest = load_tts_manifest(paths["tts_manifest"])
    caption_manifest = load_caption_manifest(paths["caption_manifest"])

    if metadata.filename_slug != Path(paths["audio_dir"]).name:
        raise ValueError("Metadata slug does not match the audio directory.")

    if metadata.filename_slug != Path(paths["captions_dir"]).name:
        raise ValueError("Metadata slug does not match the caption directory.")

    required_duration = round(
        tts_manifest.actual_duration_seconds + duration_extra_seconds,
        3,
    )

    background_probe = probe_video(background_video_path)

    selected_start = choose_background_start(
        background_duration=background_probe.duration_seconds,
        required_duration=required_duration,
        start_padding_seconds=start_padding_seconds,
        end_padding_seconds=end_padding_seconds,
        seed=random_seed,
    )

    output_directory = output_root / metadata.filename_slug
    output_directory.mkdir(parents=True, exist_ok=True)

    output_video_path = output_directory / "final.mp4"
    render_manifest_path = output_directory / "render_manifest.json"

    video_filter = build_video_filter(
        captions_filename=paths["captions_ass"].name,
        output_width=output_width,
        output_height=output_height,
        crop_anchor_y=crop_anchor_y,
    )

    command = [
        ffmpeg,
        "-y",
    ]

    if loop_background:
        command.extend(["-stream_loop", "-1"])

    if selected_start > 0:
        command.extend(["-ss", str(selected_start)])

    command.extend(
        [
            "-i",
            str(background_video_path),
            "-i",
            str(paths["narration"]),
            "-t",
            str(required_duration),
            "-filter_complex",
            (
                f"[0:v]{video_filter}[v];"
                "[1:a]loudnorm=I=-16:TP=-1.5:LRA=11[a]"
            ),
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-movflags",
            "+faststart",
            str(output_video_path),
        ]
    )

    completed = subprocess.run(
        command,
        cwd=paths["captions_ass"].parent,
        text=True,
        capture_output=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "FFmpeg render failed.\n\nSTDOUT:\n"
            f"{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )

    manifest = RenderManifest(
        source_topic_title=metadata.source_topic_title,
        video_slug=metadata.filename_slug,
        background_video=str(background_video_path),
        background_probe=background_probe,
        selected_background_start_seconds=selected_start,
        required_duration_seconds=required_duration,
        narration_audio=str(paths["narration"]),
        captions_ass=str(paths["captions_ass"]),
        metadata_file=str(paths["metadata"]),
        tts_manifest_file=str(paths["tts_manifest"]),
        caption_manifest_file=str(paths["caption_manifest"]),
        output_directory=str(output_directory),
        output_video_filename=output_video_path.name,
        output_width=output_width,
        output_height=output_height,
        crop_anchor_y=crop_anchor_y,
        loop_background=loop_background,
        ffmpeg_command=command,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    render_manifest_path.write_text(
        json.dumps(
            manifest.model_dump(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return manifest


def summarize_render_manifest(
    manifest: RenderManifest,
) -> dict[str, object]:
    return {
        "topic": manifest.source_topic_title,
        "slug": manifest.video_slug,
        "background_duration": manifest.background_probe.duration_seconds,
        "background_size": (
            f"{manifest.background_probe.width}x"
            f"{manifest.background_probe.height}"
        ),
        "selected_start": manifest.selected_background_start_seconds,
        "required_duration": manifest.required_duration_seconds,
        "output_size": f"{manifest.output_width}x{manifest.output_height}",
        "crop_anchor_y": manifest.crop_anchor_y,
        "loop_background": manifest.loop_background,
        "output_directory": manifest.output_directory,
        "output_video": manifest.output_video_filename,
    }
