"""Optimized end-to-end batch pipeline for Educational Shorts.

This module keeps the existing project files intact while removing the largest
sources of wasted time in Notebook 12:

- reuse saved outline/script artifacts when retrying a failed topic;
- skip the separate script-editor LLM call by default;
- cache extracted claims;
- retrieve evidence for multiple claims concurrently;
- verify claims and rewrite the script in one structured Ollama request;
- accept a structurally valid corrected script even when its word count misses
  the target, rather than discarding a long run;
- print start/end timing for every expensive stage.

After testing, this module can be merged into educational_shorts.batch.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, Field

from educational_shorts.batch import (
    BatchConfig,
    BatchRunManifest,
    PipelineItemResult,
    TopicBatchPlan,
    preflight_batch_pipeline,
    resolve_project_directories,
    summarize_batch_manifest,
)
from educational_shorts.captions import generate_caption_assets
from educational_shorts.client import ask_llm
from educational_shorts.editor import (
    build_edited_script_filename,
    edit_script,
    save_script as save_edited_script,
)
from educational_shorts.fact_checker import (
    _claim_quote_coverage,
    _normalize_quote,
    build_checked_script_filename,
    build_fact_check_report,
    build_report_filename,
    extract_claims,
    normalize_script,
    retrieve_evidence_for_claim,
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
    ClaimEvidenceBundle,
    ClaimVerification,
    FactualClaim,
    VideoOutline,
    VideoScript,
    VideoTopic,
)
from educational_shorts.scripts import (
    build_script_filename,
    generate_script,
    save_script as save_generated_script,
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


class OptimizedBatchConfig(BatchConfig):
    """Batch settings focused on reducing local-model runtime."""

    # One fewer full Ollama writing pass on fresh topics.
    skip_script_editor: bool = True

    # Reuse files saved before a previous failure.
    reuse_existing_intermediates: bool = True

    # Fact-check workload.
    fact_check_max_claims: int = Field(default=4, ge=1, le=10)
    fact_check_max_search_results: int = Field(default=5, ge=1, le=20)
    fact_check_max_sources_per_claim: int = Field(default=2, ge=1, le=5)
    fact_check_max_excerpt_chars: int = Field(
        default=700,
        ge=200,
        le=3000,
    )
    fact_check_cache_ttl_days: int = Field(default=90, ge=1)
    fact_check_retrieval_workers: int = Field(default=4, ge=1, le=8)

    # This is a target, not a reason to throw away a completed run.
    fact_check_minimum_words: int = Field(default=85, ge=1)
    fact_check_maximum_words: int = Field(default=150, ge=1)

    # Claim and combined-decision caches make retries much faster.
    use_claim_cache: bool = True
    use_fact_check_decision_cache: bool = True
    force_refresh_fact_check: bool = False


class CombinedFactCheckResult(BaseModel):
    """One model response containing verdicts and the final script."""

    verifications: list[ClaimVerification] = Field(default_factory=list)
    corrected_script: VideoScript


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "batch"


@contextmanager
def _stage_timer(
    label: str,
    timings: dict[str, float],
) -> Iterator[None]:
    """Print immediate progress and record elapsed seconds."""
    print(f"[START] {label}", flush=True)
    started = time.perf_counter()

    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started
        timings[label] = elapsed
        print(
            f"[FAILED] {label}: {elapsed / 60:.1f} min",
            flush=True,
        )
        raise
    else:
        elapsed = time.perf_counter() - started
        timings[label] = elapsed
        print(
            f"[DONE] {label}: {elapsed / 60:.1f} min",
            flush=True,
        )


def _print_timing_summary(timings: dict[str, float]) -> None:
    if not timings:
        return

    print()
    print("Stage timing:")
    for label, seconds in timings.items():
        print(f"  {label}: {seconds / 60:.1f} min")

    total = sum(timings.values())
    print(f"  measured total: {total / 60:.1f} min")


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
    plan: TopicBatchPlan,
    selected_topic_indexes: list[int],
) -> list[int]:
    if not selected_topic_indexes:
        raise ValueError(
            "selected_topic_indexes cannot be empty."
        )

    output: list[int] = []
    seen: set[int] = set()

    for index in selected_topic_indexes:
        if index in seen:
            continue

        if index < 0 or index >= len(plan.topics.topics):
            raise IndexError(
                f"Topic index {index} is outside the valid range "
                f"0 to {len(plan.topics.topics) - 1}."
            )

        output.append(index)
        seen.add(index)

    return output


def _load_json_model(path: Path, model_type):
    return model_type.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _stable_hash(payload: object) -> str:
    text = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _claim_cache_path(
    cache_root: Path,
    script: VideoScript,
    max_claims: int,
) -> Path:
    digest = _stable_hash(
        {
            "script": script.model_dump(),
            "max_claims": max_claims,
        }
    )
    return cache_root / "claims" / f"{digest}.json"


def _extract_claims_cached(
    *,
    script: VideoScript,
    system_prompt: str,
    max_claims: int,
    temperature: float,
    seed: int | None,
    cache_root: Path,
    use_cache: bool,
    force_refresh: bool,
) -> list[FactualClaim]:
    path = _claim_cache_path(cache_root, script, max_claims)

    if use_cache and not force_refresh and path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            claims = [
                FactualClaim.model_validate(item)
                for item in payload["claims"]
            ]
            print(
                f"[CACHE] Reused {len(claims)} extracted claims.",
                flush=True,
            )
            return claims
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            pass

    claims = extract_claims(
        script=script,
        system_prompt=system_prompt,
        max_claims=max_claims,
        temperature=temperature,
        seed=seed,
    )

    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "claims": [
                        claim.model_dump() for claim in claims
                    ]
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return claims


def _retrieve_evidence_parallel(
    *,
    claims: list[FactualClaim],
    cache_directory: Path,
    max_search_results: int,
    max_sources_per_claim: int,
    max_excerpt_chars: int,
    cache_ttl_days: int,
    force_refresh: bool,
    max_workers: int,
) -> list[ClaimEvidenceBundle]:
    """Retrieve separate claims concurrently while preserving order."""
    if not claims:
        return []

    workers = min(max_workers, len(claims))
    output: list[ClaimEvidenceBundle | None] = [None] * len(claims)

    def retrieve(index: int, claim: FactualClaim):
        bundle = retrieve_evidence_for_claim(
            claim=claim,
            cache_directory=cache_directory,
            max_search_results=max_search_results,
            max_sources=max_sources_per_claim,
            max_excerpt_chars=max_excerpt_chars,
            cache_ttl_days=cache_ttl_days,
            force_refresh=force_refresh,
        )
        return index, bundle

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(retrieve, index, claim)
            for index, claim in enumerate(claims)
        ]

        for future in as_completed(futures):
            index, bundle = future.result()
            output[index] = bundle
            source_label = "cache" if bundle.used_cache else "web"
            print(
                f"  Retrieved {index + 1}/{len(claims)}: "
                f"{len(bundle.sources)} source(s) from {source_label}",
                flush=True,
            )

    return [
        bundle
        for bundle in output
        if bundle is not None
    ]


def _insufficient_verification(
    bundle: ClaimEvidenceBundle,
    explanation: str,
) -> ClaimVerification:
    return ClaimVerification(
        claim_id=bundle.claim.claim_id,
        verdict="insufficient_evidence",
        severity="medium",
        explanation=explanation,
        correction=None,
        supporting_urls=[],
        evidence_quote=None,
        evidence_url=None,
    )


def _normalize_verifications(
    raw_verifications: list[ClaimVerification],
    evidence_bundles: list[ClaimEvidenceBundle],
) -> list[ClaimVerification]:
    """Keep model verdicts tied to the evidence actually supplied."""
    bundle_by_id = {
        bundle.claim.claim_id: bundle
        for bundle in evidence_bundles
    }
    output: list[ClaimVerification] = []
    seen: set[str] = set()

    for verification in raw_verifications:
        bundle = bundle_by_id.get(verification.claim_id)

        if bundle is None or verification.claim_id in seen:
            continue

        allowed_urls = {source.url for source in bundle.sources}
        verification.supporting_urls = [
            url
            for url in verification.supporting_urls
            if url in allowed_urls
        ]

        if not bundle.sources:
            verification = _insufficient_verification(
                bundle,
                bundle.retrieval_error or "No evidence was available.",
            )

        elif verification.verdict == "supported":
            evidence_quote = (
                verification.evidence_quote or ""
            ).strip()
            matching_source = None

            for source in bundle.sources:
                normalized_quote = _normalize_quote(evidence_quote)
                normalized_excerpt = _normalize_quote(source.excerpt)

                if (
                    normalized_quote
                    and normalized_quote in normalized_excerpt
                ):
                    matching_source = source
                    break

            if matching_source is None:
                verification = _insufficient_verification(
                    bundle,
                    (
                        "The verifier marked this claim as supported but "
                        "did not provide a valid quote copied from the "
                        "supplied evidence."
                    ),
                )
            else:
                coverage = _claim_quote_coverage(
                    bundle.claim.atomic_claim,
                    evidence_quote,
                )

                if coverage < 0.55:
                    verification = _insufficient_verification(
                        bundle,
                        (
                            "The evidence quote exists, but it does not "
                            "cover enough of the claim. Lexical coverage: "
                            f"{coverage:.0%}."
                        ),
                    )
                else:
                    verification.severity = "none"
                    verification.correction = None
                    verification.evidence_url = matching_source.url
                    verification.supporting_urls = [
                        matching_source.url
                    ]

        output.append(verification)
        seen.add(verification.claim_id)

    for bundle in evidence_bundles:
        claim_id = bundle.claim.claim_id

        if claim_id not in seen:
            output.append(
                _insufficient_verification(
                    bundle,
                    bundle.retrieval_error
                    or "The verifier omitted this claim.",
                )
            )

    order = {
        bundle.claim.claim_id: index
        for index, bundle in enumerate(evidence_bundles)
    }
    return sorted(output, key=lambda item: order[item.claim_id])


def _build_combined_fact_check_prompt(
    *,
    script: VideoScript,
    evidence_bundles: list[ClaimEvidenceBundle],
    minimum_words: int,
    maximum_words: int,
) -> str:
    payload = [
        {
            "claim": bundle.claim.model_dump(),
            "sources": [
                source.model_dump() for source in bundle.sources
            ],
            "retrieval_error": bundle.retrieval_error,
        }
        for bundle in evidence_bundles
    ]

    return f"""
Perform the verification and correction in one operation.

SOURCE SCRIPT:
{json.dumps(script.model_dump(), indent=2, ensure_ascii=False)}

CLAIMS AND EVIDENCE:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Return:
1. exactly one ClaimVerification for every supplied claim_id; and
2. a corrected VideoScript ready for narration.

Verification rules:
- Use only the evidence supplied for that claim.
- supported means the evidence directly supports every material part.
- contradicted means the evidence directly conflicts with the wording.
- overstated means the basic idea is plausible but too broad or too certain.
- insufficient_evidence means the passages do not establish a verdict.
- For supported, copy one exact evidence_quote from the supplied excerpts.
- evidence_url and supporting_urls may contain only supplied URLs.
- For contradicted or overstated, provide a concise evidence-based correction.
- Never fill an evidence gap from memory.

Script rules:
- Preserve the exact topic object.
- Preserve the script's spoken order and general structure.
- Apply the corrections indicated by your own verifications.
- For insufficient evidence, naturally soften or remove the unsupported part.
- Do not invent facts, studies, numbers, names, dates, or mechanisms.
- Do not mention evidence, sources, fact checking, or uncertainty in the
  research process inside the narration.
- Do not include citations, URLs, headings, visual directions, hashtags, or
  engagement requests in narration.
- Aim for {minimum_words} to {maximum_words} total words.
- Word count is a target. Accuracy and a complete natural script matter more
  than adding filler to reach the minimum.
- Return only structured output matching the requested schema.
""".strip()


def _combined_decision_cache_path(
    cache_root: Path,
    script: VideoScript,
    evidence_bundles: list[ClaimEvidenceBundle],
    minimum_words: int,
    maximum_words: int,
) -> Path:
    digest = _stable_hash(
        {
            "script": script.model_dump(),
            "evidence": [
                bundle.model_dump()
                for bundle in evidence_bundles
            ],
            "minimum_words": minimum_words,
            "maximum_words": maximum_words,
            "pipeline_version": 1,
        }
    )
    return (
        cache_root
        / "combined_fact_checks"
        / f"{digest}.json"
    )


def _verify_and_rewrite_once(
    *,
    script: VideoScript,
    evidence_bundles: list[ClaimEvidenceBundle],
    system_prompt: str,
    target_wpm: int,
    minimum_words: int,
    maximum_words: int,
    temperature: float,
    seed: int | None,
    cache_root: Path,
    use_cache: bool,
    force_refresh: bool,
) -> tuple[list[ClaimVerification], VideoScript]:
    """Use one Ollama response for verdicts and the corrected script."""
    if not evidence_bundles:
        return [], normalize_script(
            script.model_copy(deep=True),
            target_wpm,
        )

    cache_path = _combined_decision_cache_path(
        cache_root,
        script,
        evidence_bundles,
        minimum_words,
        maximum_words,
    )

    result: CombinedFactCheckResult | None = None

    if use_cache and not force_refresh and cache_path.exists():
        try:
            result = CombinedFactCheckResult.model_validate_json(
                cache_path.read_text(encoding="utf-8")
            )
            print(
                "[CACHE] Reused combined fact-check decision.",
                flush=True,
            )
        except (OSError, ValueError):
            result = None

    if result is None:
        result = ask_llm(
            system_prompt=system_prompt,
            user_prompt=_build_combined_fact_check_prompt(
                script=script,
                evidence_bundles=evidence_bundles,
                minimum_words=minimum_words,
                maximum_words=maximum_words,
            ),
            schema=CombinedFactCheckResult,
            temperature=temperature,
            seed=seed,
        )

        if use_cache:
            cache_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            cache_path.write_text(
                result.model_dump_json(indent=2),
                encoding="utf-8",
            )

    verifications = _normalize_verifications(
        raw_verifications=result.verifications,
        evidence_bundles=evidence_bundles,
    )

    corrected = result.corrected_script
    corrected.topic = script.topic
    corrected = normalize_script(
        corrected,
        words_per_minute=target_wpm,
    )

    if len(corrected.sections) != len(script.sections):
        print(
            "[WARNING] Corrected script changed the body-section count "
            f"from {len(script.sections)} to {len(corrected.sections)}. "
            "The script will still be used.",
            flush=True,
        )

    if corrected.word_count < minimum_words:
        print(
            "[WARNING] Corrected script contains "
            f"{corrected.word_count} words, below the "
            f"{minimum_words}-word target. It will be accepted rather "
            "than discarding the completed run.",
            flush=True,
        )
    elif corrected.word_count > maximum_words:
        print(
            "[WARNING] Corrected script contains "
            f"{corrected.word_count} words, above the "
            f"{maximum_words}-word target. It will be accepted rather "
            "than making another slow model call.",
            flush=True,
        )

    return verifications, corrected


def _load_or_generate_outline(
    *,
    topic: VideoTopic,
    path: Path,
    config: OptimizedBatchConfig,
    system_prompt: str,
    seed: int,
) -> VideoOutline:
    if config.reuse_existing_intermediates and path.exists():
        print(f"[REUSE] Outline: {path}", flush=True)
        return _load_json_model(path, VideoOutline)

    outline = generate_outline(
        topic=topic,
        system_prompt=system_prompt,
        target_seconds=config.target_seconds,
        section_count=config.section_count,
        temperature=config.outline_temperature,
        seed=seed,
    )
    save_outline(outline=outline, output_path=path)
    return outline


def _load_or_generate_script(
    *,
    outline: VideoOutline,
    path: Path,
    config: OptimizedBatchConfig,
    system_prompt: str,
    seed: int,
) -> VideoScript:
    if config.reuse_existing_intermediates and path.exists():
        print(f"[REUSE] Script: {path}", flush=True)
        return _load_json_model(path, VideoScript)

    script = generate_script(
        outline=outline,
        system_prompt=system_prompt,
        target_wpm=config.target_words_per_minute,
        temperature=config.script_temperature,
        seed=seed,
    )
    save_generated_script(script=script, output_path=path)
    return script


def _load_or_edit_script(
    *,
    script: VideoScript,
    path: Path,
    config: OptimizedBatchConfig,
    system_prompt: str,
    seed: int,
) -> tuple[VideoScript, str]:
    if config.reuse_existing_intermediates and path.exists():
        print(f"[REUSE] Edited script: {path}", flush=True)
        return _load_json_model(path, VideoScript), "script_editing_reused"

    if config.skip_script_editor:
        edited_script = normalize_script(
            script.model_copy(deep=True),
            config.target_words_per_minute,
        )
        save_edited_script(
            script=edited_script,
            output_path=path,
        )
        print(
            "[SKIP] Separate script-editor Ollama call disabled.",
            flush=True,
        )
        return edited_script, "script_editing_skipped"

    edited_script = edit_script(
        script=script,
        system_prompt=system_prompt,
        target_wpm=config.target_words_per_minute,
        minimum_seconds=config.editor_minimum_seconds,
        maximum_seconds=config.editor_maximum_seconds,
        temperature=config.editor_temperature,
        seed=seed,
    )
    save_edited_script(
        script=edited_script,
        output_path=path,
    )
    return edited_script, "script_editing"


def _run_one_topic_optimized(
    *,
    project_root: Path,
    config: OptimizedBatchConfig,
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
    timings: dict[str, float] = {}
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

        outline_path = (
            directories["outlines"]
            / build_outline_filename(topic)
        )

        stage = "outline_generation"
        with _stage_timer("Outline generation", timings):
            outline = _load_or_generate_outline(
                topic=topic,
                path=outline_path,
                config=config,
                system_prompt=prompts["outline"],
                seed=config.outline_seed + seed_offset,
            )
        paths["outline"] = str(outline_path)
        stages_completed.append(stage)

        script_path = (
            directories["scripts"]
            / build_script_filename(outline)
        )

        stage = "script_generation"
        with _stage_timer("Script generation", timings):
            script = _load_or_generate_script(
                outline=outline,
                path=script_path,
                config=config,
                system_prompt=prompts["script"],
                seed=config.script_seed + seed_offset,
            )
        paths["script"] = str(script_path)
        stages_completed.append(stage)

        edited_script_path = (
            directories["edited_scripts"]
            / build_edited_script_filename(script)
        )

        stage = "script_editing"
        with _stage_timer("Script editing", timings):
            edited_script, editor_stage = _load_or_edit_script(
                script=script,
                path=edited_script_path,
                config=config,
                system_prompt=prompts["editor"],
                seed=config.editor_seed + seed_offset,
            )
        paths["edited_script"] = str(edited_script_path)
        stages_completed.append(editor_stage)

        cache_root = directories["data"] / "retrieval_cache"

        stage = "claim_extraction"
        with _stage_timer("Claim extraction", timings):
            claims = _extract_claims_cached(
                script=edited_script,
                system_prompt=prompts["fact_checker"],
                max_claims=config.fact_check_max_claims,
                temperature=config.fact_check_temperature,
                seed=config.fact_check_seed + seed_offset,
                cache_root=cache_root,
                use_cache=config.use_claim_cache,
                force_refresh=config.force_refresh_fact_check,
            )
        stages_completed.append(stage)
        print(
            f"Claims selected for checking: {len(claims)}",
            flush=True,
        )

        stage = "evidence_retrieval"
        with _stage_timer("Evidence retrieval", timings):
            evidence_bundles = _retrieve_evidence_parallel(
                claims=claims,
                cache_directory=cache_root,
                max_search_results=(
                    config.fact_check_max_search_results
                ),
                max_sources_per_claim=(
                    config.fact_check_max_sources_per_claim
                ),
                max_excerpt_chars=(
                    config.fact_check_max_excerpt_chars
                ),
                cache_ttl_days=(
                    config.fact_check_cache_ttl_days
                ),
                force_refresh=config.force_refresh_fact_check,
                max_workers=(
                    config.fact_check_retrieval_workers
                ),
            )
        stages_completed.append(stage)

        stage = "fact_checking"
        with _stage_timer(
            "Combined verification and rewrite",
            timings,
        ):
            verifications, corrected_script = (
                _verify_and_rewrite_once(
                    script=edited_script,
                    evidence_bundles=evidence_bundles,
                    system_prompt=prompts["fact_checker"],
                    target_wpm=config.target_words_per_minute,
                    minimum_words=(
                        config.fact_check_minimum_words
                    ),
                    maximum_words=(
                        config.fact_check_maximum_words
                    ),
                    temperature=0.0,
                    seed=config.fact_check_seed + seed_offset,
                    cache_root=cache_root,
                    use_cache=(
                        config.use_fact_check_decision_cache
                    ),
                    force_refresh=(
                        config.force_refresh_fact_check
                    ),
                )
            )

        report = build_fact_check_report(
            claims=claims,
            evidence_bundles=evidence_bundles,
            verifications=verifications,
            corrected_script=corrected_script,
        )

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
        save_report(report=report, output_path=report_path)
        paths["checked_script"] = str(checked_script_path)
        paths["fact_check_report"] = str(report_path)
        stages_completed.append(stage)

        if (
            report.requires_manual_review
            and config.pause_on_manual_review
        ):
            _print_timing_summary(timings)
            return (
                PipelineItemResult(
                    item_number=item_number,
                    topic_index=topic_index,
                    topic_title=topic.title,
                    status="manual_review",
                    final_stage=stage,
                    message=(
                        "Fact checking completed, but the topic was held "
                        "before publication assets because manual review "
                        "is enabled."
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
        with _stage_timer("Metadata generation", timings):
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
        with _stage_timer("Kokoro TTS", timings):
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
                trim_edge_silence=(
                    config.trim_tts_edge_silence
                ),
                target_peak_dbfs=config.tts_target_peak_dbfs,
            )
        paths["tts_manifest"] = str(tts_result.manifest_path)
        paths["narration"] = str(tts_result.master_audio_path)
        paths["transcript"] = str(tts_result.transcript_path)
        stages_completed.append(stage)

        stage = "caption_generation"
        with _stage_timer("Caption generation", timings):
            caption_manifest = generate_caption_assets(
                tts_manifest=tts_result.manifest,
                source_tts_manifest_path=(
                    tts_result.manifest_path
                ),
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

        caption_directory = Path(
            caption_manifest.output_directory
        )
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
        with _stage_timer("Video assembly", timings):
            render_manifest = render_video(
                project_root=project_root,
                metadata_path=metadata_path,
                background_video_path=background_video_path,
                output_root=directories["videos"],
                output_width=config.output_width,
                output_height=config.output_height,
                crop_anchor_y=config.crop_anchor_y,
                random_seed=(
                    config.video_random_seed + seed_offset
                ),
                duration_extra_seconds=(
                    config.duration_extra_seconds
                ),
                start_padding_seconds=(
                    config.start_padding_seconds
                ),
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

        _print_timing_summary(timings)

        return (
            PipelineItemResult(
                item_number=item_number,
                topic_index=topic_index,
                topic_title=topic.title,
                status="completed",
                final_stage=stage,
                message=(
                    "Finished the optimized educational-short pipeline."
                ),
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
        _print_timing_summary(timings)
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


def run_batch_pipeline_optimized(
    *,
    project_root: Path,
    plan: TopicBatchPlan,
    selected_topic_indexes: list[int],
    config: OptimizedBatchConfig,
) -> BatchRunManifest:
    """Run selected topics through the optimized pipeline."""
    indexes = _validate_selected_indexes(
        plan,
        selected_topic_indexes,
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

    find_ffmpeg_executable()
    find_ffprobe_executable()
    prompts = _load_pipeline_prompts()

    run_id = _run_id()
    output_directory = directories["batch_runs"] / run_id
    manifest_path = output_directory / "batch_manifest.json"

    manifest = BatchRunManifest(
        marker="NOTEBOOK_12_BATCH_PIPELINE_OPTIMIZED",
        run_id=run_id,
        category_path=plan.category_path,
        selected_topic_indexes=indexes,
        selected_topic_titles=[
            topic.title for topic in selected_topics
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

    print(
        "Optimized pipeline enabled: "
        f"editor skipped={config.skip_script_editor}, "
        f"max claims={config.fact_check_max_claims}, "
        "verification and rewrite combined into one model call.",
        flush=True,
    )

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

        result, synthesizer = _run_one_topic_optimized(
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


__all__ = [
    "OptimizedBatchConfig",
    "TopicBatchPlan",
    "preflight_batch_pipeline",
    "run_batch_pipeline_optimized",
    "summarize_batch_manifest",
]
