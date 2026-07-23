"""WEB_RAG_FACT_CHECKER_FRESH_V3

Retrieval-grounded fact checking for the Educational Shorts project.

This module intentionally does NOT define the old fact_check_script() function
and never uses report.issues. It works with the retrieval-grounded schemas:
FactualClaim, ClaimEvidenceBundle, ClaimVerification, and FactCheckReport.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
import trafilatura
from ddgs import DDGS

from educational_shorts.client import ask_llm
from educational_shorts.schemas import (
    ClaimEvidenceBundle,
    ClaimList,
    ClaimVerification,
    ClaimVerificationList,
    EvidenceSource,
    FactCheckReport,
    FactualClaim,
    VideoScript,
)


USER_AGENT = "EducationalShortsWebRAG/3.0 (local research pipeline)"

TRUSTED_DOMAIN_SCORES = {
    "nih.gov": 14,
    "ncbi.nlm.nih.gov": 14,
    "cdc.gov": 14,
    "fda.gov": 13,
    "nasa.gov": 13,
    "noaa.gov": 13,
    "usgs.gov": 13,
    "who.int": 13,
    "si.edu": 11,
    "nature.com": 10,
    "science.org": 10,
    "pnas.org": 10,
    "cell.com": 9,
    "britannica.com": 8,
    "wikipedia.org": 4,
}

BLOCKED_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "pinterest.com",
    "quora.com",
    "reddit.com",
    "tiktok.com",
    "x.com",
    "youtube.com",
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "if", "in", "into", "is", "it", "its", "may", "might",
    "of", "on", "or", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "would",
}


def find_edited_script_file(
    scripts_directory: Path,
    filename: str | None = None,
) -> Path:
    """Use an explicit edited script or the newest JSON file."""
    if filename:
        path = scripts_directory / filename
        if not path.exists():
            raise FileNotFoundError(f"Edited script not found: {path}")
        return path

    candidates = sorted(
        scripts_directory.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No edited script JSON files found in {scripts_directory}."
        )
    return candidates[0]


def load_script(path: Path) -> VideoScript:
    return VideoScript.model_validate_json(path.read_text(encoding="utf-8"))


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text))


def estimate_seconds(text: str, words_per_minute: int = 145) -> int:
    if words_per_minute < 1:
        raise ValueError("words_per_minute must be positive.")
    return max(1, round(count_words(text) * 60 / words_per_minute))


def normalize_script(
    script: VideoScript,
    words_per_minute: int = 145,
) -> VideoScript:
    """Recalculate segment timing, full narration, and totals."""
    segments = [script.hook, *script.sections, script.closing]

    for segment in segments:
        segment.narration = " ".join(segment.narration.split())
        segment.estimated_seconds = estimate_seconds(
            segment.narration,
            words_per_minute,
        )

    script.full_narration = "\n\n".join(
        segment.narration for segment in segments
    )
    script.word_count = count_words(script.full_narration)
    script.estimated_total_seconds = sum(
        segment.estimated_seconds for segment in segments
    )
    return script


def build_claim_extraction_prompt(
    script: VideoScript,
    max_claims: int,
) -> str:
    return f"""
Extract the factual claims from this short narrated script that most need
external verification.

SCRIPT:
{json.dumps(script.model_dump(), indent=2, ensure_ascii=False)}

Rules:
- Return at most {max_claims} claims.
- Prioritize definitions, mechanisms, medicine, causation, dates, numbers,
  named discoveries, superlatives, and broad generalizations.
- Ignore rhetorical or subjective lines.
- Make each atomic_claim independently verifiable.
- original_text must quote the relevant wording from the script.
- search_query must be concise and useful for web search.
- Use claim IDs claim_1, claim_2, and so on.
- Do not decide whether the claims are true yet.
- Return only structured output matching the schema.
""".strip()


def extract_claims(
    script: VideoScript,
    system_prompt: str,
    max_claims: int = 10,
    temperature: float = 0.1,
    seed: int | None = None,
) -> list[FactualClaim]:
    result = ask_llm(
        system_prompt=system_prompt,
        user_prompt=build_claim_extraction_prompt(script, max_claims),
        schema=ClaimList,
        temperature=temperature,
        seed=seed,
    )

    claims: list[FactualClaim] = []
    seen: set[str] = set()

    for candidate in result.claims[:max_claims]:
        key = re.sub(r"\W+", " ", candidate.atomic_claim.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        candidate.claim_id = f"claim_{len(claims) + 1}"
        candidate.search_query = " ".join(candidate.search_query.split())
        claims.append(candidate)

    return claims


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }

def _claim_quote_coverage(
    claim_text: str,
    evidence_quote: str,
) -> float:
    """Estimate how much of a claim is represented in its evidence quote."""
    claim_terms = _tokens(claim_text)
    quote_terms = _tokens(evidence_quote)

    if not claim_terms:
        return 0.0

    return len(claim_terms & quote_terms) / len(claim_terms)


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _matches_domain(domain: str, candidate: str) -> bool:
    return domain == candidate or domain.endswith("." + candidate)


def _domain_score(domain: str) -> int:
    if any(_matches_domain(domain, item) for item in BLOCKED_DOMAINS):
        return -50

    score = 0
    for trusted, value in TRUSTED_DOMAIN_SCORES.items():
        if _matches_domain(domain, trusted):
            score = max(score, value)

    if domain.endswith(".gov"):
        score = max(score, 12)
    elif domain.endswith(".edu"):
        score = max(score, 10)
    elif domain.endswith(".org"):
        score = max(score, 3)

    return score


def _result_score(result: dict, query: str) -> int:
    url = result.get("href") or result.get("url") or ""
    title = result.get("title") or ""
    snippet = result.get("body") or result.get("snippet") or ""
    overlap = len(_tokens(query) & _tokens(f"{title} {snippet}"))
    return _domain_score(_domain(url)) + overlap * 2


def _clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fetch_main_text(url: str, timeout_seconds: int = 12) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.path.lower().endswith(".pdf"):
        return None

    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout_seconds,
        allow_redirects=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if not any(
        allowed in content_type
        for allowed in ("text/html", "application/xhtml+xml", "text/plain")
    ):
        return None

    extracted = trafilatura.extract(
        response.text,
        url=str(response.url),
        include_comments=False,
        include_tables=False,
        favor_precision=True,
        deduplicate=True,
    )
    return _clean_text(extracted) if extracted else None


def _best_excerpt(
    text: str,
    claim: FactualClaim,
    max_chars: int,
) -> str:
    text = _clean_text(text)
    if len(text) <= max_chars:
        return text

    target_terms = _tokens(
        f"{claim.atomic_claim} {claim.search_query}"
    )
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n{2,}", text)
        if len(paragraph.strip()) >= 60
    ]

    if not paragraphs:
        return text[:max_chars].rsplit(" ", 1)[0]

    ranked = sorted(
        paragraphs,
        key=lambda paragraph: len(target_terms & _tokens(paragraph)),
        reverse=True,
    )
    excerpt = "\n\n".join(ranked[:3])[:max_chars]
    return excerpt.rsplit(" ", 1)[0] if len(excerpt) == max_chars else excerpt


def _cache_path(cache_directory: Path, query: str) -> Path:
    normalized = " ".join(query.lower().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return cache_directory / f"{digest}.json"


def _load_cache(
    cache_directory: Path,
    query: str,
    ttl_days: int,
) -> list[EvidenceSource] | None:
    path = _cache_path(cache_directory, query)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cached_at = datetime.fromisoformat(payload["cached_at"])
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)

        if datetime.now(timezone.utc) > cached_at + timedelta(days=ttl_days):
            return None

        return [
            EvidenceSource.model_validate(item)
            for item in payload.get("sources", [])
        ]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _save_cache(
    cache_directory: Path,
    query: str,
    sources: list[EvidenceSource],
) -> None:
    cache_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": query,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "sources": [source.model_dump() for source in sources],
    }
    _cache_path(cache_directory, query).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def retrieve_evidence_for_claim(
    claim: FactualClaim,
    cache_directory: Path,
    max_search_results: int = 8,
    max_sources: int = 3,
    max_excerpt_chars: int = 1000,
    cache_ttl_days: int = 30,
    force_refresh: bool = False,
) -> ClaimEvidenceBundle:
    """Retrieve a few short evidence passages for one claim."""
    if not force_refresh:
        cached = _load_cache(
            cache_directory,
            claim.search_query,
            cache_ttl_days,
        )
        if cached is not None:
            return ClaimEvidenceBundle(
                claim=claim,
                sources=cached,
                used_cache=True,
            )

    try:
        results = list(
            DDGS(timeout=12).text(
                claim.search_query,
                region="us-en",
                safesearch="moderate",
                max_results=max_search_results,
                backend="auto",
            )
        )
    except Exception as exc:
        return ClaimEvidenceBundle(
            claim=claim,
            sources=[],
            retrieval_error=f"Search failed: {exc}",
        )

    results.sort(
        key=lambda result: _result_score(result, claim.search_query),
        reverse=True,
    )

    sources: list[EvidenceSource] = []
    seen_urls: set[str] = set()
    retrieved_at = datetime.now(timezone.utc).isoformat()

    for result in results[: max_sources * 4]:
        url = result.get("href") or result.get("url") or ""
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        domain = _domain(url)
        if _domain_score(domain) < 0:
            continue

        title = " ".join(
            (result.get("title") or domain or "Untitled source").split()
        )
        snippet = _clean_text(
            result.get("body") or result.get("snippet") or ""
        )

        try:
            page_text = _fetch_main_text(url)
        except requests.RequestException:
            page_text = None

        source_text = page_text or snippet
        if len(source_text) < 80:
            continue

        excerpt = _best_excerpt(
            source_text,
            claim,
            max_excerpt_chars,
        )
        if len(excerpt) < 60:
            continue

        sources.append(
            EvidenceSource(
                title=title,
                url=url,
                domain=domain,
                excerpt=excerpt,
                source_score=_result_score(result, claim.search_query),
                retrieved_at=retrieved_at,
            )
        )
        if len(sources) >= max_sources:
            break

    _save_cache(cache_directory, claim.search_query, sources)

    return ClaimEvidenceBundle(
        claim=claim,
        sources=sources,
        used_cache=False,
        retrieval_error=(
            None
            if sources
            else "Search completed but no usable evidence was extracted."
        ),
    )


def retrieve_evidence(
    claims: list[FactualClaim],
    cache_directory: Path,
    max_search_results: int = 8,
    max_sources_per_claim: int = 3,
    max_excerpt_chars: int = 1000,
    cache_ttl_days: int = 30,
    force_refresh: bool = False,
) -> list[ClaimEvidenceBundle]:
    bundles = []

    for index, claim in enumerate(claims, start=1):
        print(
            f"Retrieving {index}/{len(claims)}: {claim.atomic_claim}"
        )
        bundle = retrieve_evidence_for_claim(
            claim=claim,
            cache_directory=cache_directory,
            max_search_results=max_search_results,
            max_sources=max_sources_per_claim,
            max_excerpt_chars=max_excerpt_chars,
            cache_ttl_days=cache_ttl_days,
            force_refresh=force_refresh,
        )
        bundles.append(bundle)
        source_label = "cache" if bundle.used_cache else "web"
        print(f"  {len(bundle.sources)} source(s) from {source_label}")

    return bundles


def build_verification_prompt(
    bundles: list[ClaimEvidenceBundle],
) -> str:
    payload = [
        {
            "claim": bundle.claim.model_dump(),
            "sources": [
                source.model_dump() for source in bundle.sources
            ],
            "retrieval_error": bundle.retrieval_error,
        }
        for bundle in bundles
    ]

    return f"""
Judge each claim using only its supplied evidence.

CLAIMS AND EVIDENCE:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Rules:
- Return exactly one result for each claim_id.
- supported: evidence directly supports the wording.
- contradicted: evidence directly conflicts with the wording.
- overstated: the basic idea is plausible but too broad, certain, or imprecise.
- insufficient_evidence: the supplied passages do not establish a verdict.
- For contradicted or overstated claims, give a concise correction supported by
  the evidence.
- For supported claims, correction must be null.
- supporting_urls may contain only URLs supplied for that claim.
- Never fill evidence gaps from memory.
- Return only structured output matching the schema.
- Judge the exact atomic claim, not merely whether the sources discuss the
  same topic.
- Every material part of the claim must be supported.
- If one part is wrong, imprecise, or unsupported, do not use "supported".
- For "supported", provide one exact evidence_quote copied from the supplied
  excerpts and its evidence_url.
- The quote must directly entail the claim.
- Similar subject matter is not sufficient.
- If the claim incorrectly names or defines something, use "contradicted" or
  "overstated".
- If the explanation does not address the exact wording of the claim, the
  verdict cannot be "supported".
  - A quote mentioning the same subject is not enough.
- Every material part of the atomic claim must be established by the quote.
- For a supported verdict, the explanation must explicitly connect each part
  of the claim to the evidence quote.
- Claims about breakthroughs, improved health, medical applications, causation,
  importance, or impact require direct evidence for those specific assertions.
- If the quote supports only one portion of a compound claim, use overstated
  or insufficient_evidence, not supported.
""".strip()


def verify_claims(
    evidence_bundles: list[ClaimEvidenceBundle],
    system_prompt: str,
    temperature: float = 0.0,
    seed: int | None = None,
) -> list[ClaimVerification]:
    if not evidence_bundles:
        return []

    raw_verifications: list[ClaimVerification] = []

    for index, bundle in enumerate(evidence_bundles, start=1):
        print(
            f"Verifying claim {index}/{len(evidence_bundles)}: "
            f"{bundle.claim.atomic_claim}"
        )

        result = ask_llm(
            system_prompt=system_prompt,
            user_prompt=build_verification_prompt([bundle]),
            schema=ClaimVerificationList,
            temperature=temperature,
            seed=None if seed is None else seed + index - 1,
        )

        if len(result.verifications) == 1:
            raw_verifications.append(result.verifications[0])
        else:
            raw_verifications.append(
                ClaimVerification(
                    claim_id=bundle.claim.claim_id,
                    verdict="insufficient_evidence",
                    severity="medium",
                    explanation=(
                        "The verifier did not return exactly one result "
                        "for this claim."
                    ),
                    correction=None,
                    supporting_urls=[],
                    evidence_quote=None,
                    evidence_url=None,
                )
            )

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

        allowed_urls = {
            source.url for source in bundle.sources
        }

        verification.supporting_urls = [
            url
            for url in verification.supporting_urls
            if url in allowed_urls
        ]

        if not bundle.sources:
            verification.verdict = "insufficient_evidence"
            verification.severity = "medium"
            verification.correction = None
            verification.supporting_urls = []
            verification.evidence_quote = None
            verification.evidence_url = None
            verification.explanation = (
                bundle.retrieval_error
                or "No evidence was available."
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
                verification.verdict = "insufficient_evidence"
                verification.severity = "medium"
                verification.correction = None
                verification.supporting_urls = []
                verification.evidence_quote = None
                verification.evidence_url = None
                verification.explanation = (
                    "The verifier marked this claim as supported but did not "
                    "provide a valid quote copied from the supplied evidence."
                )
            else:
                coverage = _claim_quote_coverage(
                    bundle.claim.atomic_claim,
                    evidence_quote,
                )

                if coverage < 0.55:
                    verification.verdict = "insufficient_evidence"
                    verification.severity = "medium"
                    verification.correction = None
                    verification.supporting_urls = []
                    verification.evidence_url = None
                    verification.explanation = (
                        "The evidence quote exists in the retrieved source, but it "
                        "does not cover enough of the claim to support every material "
                        f"part. Lexical coverage: {coverage:.0%}."
                    )
                else:
                    verification.severity = "none"
                    verification.correction = None
                    verification.evidence_url = matching_source.url
                    verification.supporting_urls = [matching_source.url]

        output.append(verification)
        seen.add(verification.claim_id)

    for bundle in evidence_bundles:
        claim_id = bundle.claim.claim_id
        if claim_id not in seen:
            output.append(
                ClaimVerification(
                    claim_id=claim_id,
                    verdict="insufficient_evidence",
                    severity="medium",
                    explanation=(
                        bundle.retrieval_error
                        or "The verifier omitted this claim."
                    ),
                    correction=None,
                    supporting_urls=[],
                )
            )

    order = {
        bundle.claim.claim_id: index
        for index, bundle in enumerate(evidence_bundles)
    }
    return sorted(output, key=lambda item: order[item.claim_id])


def build_rewrite_prompt(
    script: VideoScript,
    bundles: list[ClaimEvidenceBundle],
    verifications: list[ClaimVerification],
    minimum_words: int,
    maximum_words: int,
) -> str:
    claim_lookup = {
        bundle.claim.claim_id: bundle.claim
        for bundle in bundles
    }

    corrections = []
    for verification in verifications:
        if verification.verdict == "supported":
            continue
        claim = claim_lookup[verification.claim_id]
        corrections.append(
            {
                "claim_id": verification.claim_id,
                "segment_type": claim.segment_type,
                "original_text": claim.original_text,
                "verdict": verification.verdict,
                "explanation": verification.explanation,
                "approved_correction": verification.correction,
            }
        )

    return f"""
Apply only these evidence-grounded corrections to the source script.

SOURCE SCRIPT:
{json.dumps(script.model_dump(), indent=2, ensure_ascii=False)}

CORRECTIONS:
{json.dumps(corrections, indent=2, ensure_ascii=False)}

Rules:
- Preserve the exact topic object.
- Preserve the number and order of body sections.
- Change only wording needed for accuracy or qualification.
- Use approved_correction when provided.
- For insufficient evidence without a correction, soften or remove the claim.
- Do not invent facts, studies, numbers, names, dates, or mechanisms.
- Do not include citations, URLs, source names, headings, visual directions,
  hashtags, or engagement requests in narration.
- Keep total narration between {minimum_words} and {maximum_words} words.
- Return only the corrected VideoScript.
- Never mention "the sources", "the evidence", "the available information",
  "the fact checker", or uncertainty in the research process inside narration.
- When evidence does not support part of a sentence, remove that part or replace
  it with a natural statement that the evidence does support.
- The corrected narration must sound like a finished educational script, not
  a fact-check report.
""".strip()


def rewrite_corrected_script(
    script: VideoScript,
    evidence_bundles: list[ClaimEvidenceBundle],
    verifications: list[ClaimVerification],
    system_prompt: str,
    target_wpm: int = 145,
    minimum_words: int = 105,
    maximum_words: int = 150,
    temperature: float = 0.2,
    seed: int | None = None,
    max_attempts: int = 3,
) -> VideoScript:
    needs_rewrite = any(
        item.verdict != "supported" for item in verifications
    )
    if not needs_rewrite:
        return normalize_script(
            script.model_copy(deep=True),
            target_wpm,
        )

    prompt = build_rewrite_prompt(
        script,
        evidence_bundles,
        verifications,
        minimum_words,
        maximum_words,
    )

    failures: list[str] = []
    attempt_prompt = prompt

    for attempt in range(1, max_attempts + 1):
        attempt_seed = None if seed is None else seed + attempt - 1

        corrected = ask_llm(
            system_prompt=system_prompt,
            user_prompt=attempt_prompt,
            schema=VideoScript,
            temperature=temperature,
            seed=attempt_seed,
        )

        if len(corrected.sections) != len(script.sections):
            feedback = (
                f"The previous attempt returned {len(corrected.sections)} body "
                f"sections, but exactly {len(script.sections)} are required."
            )
            failures.append(f"attempt {attempt}: wrong section count")

        else:
            corrected.topic = script.topic
            corrected = normalize_script(
                corrected,
                words_per_minute=target_wpm,
            )

            if corrected.word_count < minimum_words:
                words_needed = minimum_words - corrected.word_count

                feedback = (
                    f"The previous attempt contained {corrected.word_count} words, "
                    f"which is below the minimum of {minimum_words}. Add at least "
                    f"{words_needed + 5} useful words. Expand accurate explanations "
                    "or transitions without restoring unsupported claims, adding "
                    "filler, or introducing new facts."
                )

                failures.append(
                    f"attempt {attempt}: {corrected.word_count} words"
                )

            elif corrected.word_count > maximum_words:
                words_to_remove = corrected.word_count - maximum_words

                feedback = (
                    f"The previous attempt contained {corrected.word_count} words, "
                    f"which exceeds the maximum of {maximum_words}. Remove at least "
                    f"{words_to_remove + 3} words while preserving every factual "
                    "correction."
                )

                failures.append(
                    f"attempt {attempt}: {corrected.word_count} words"
                )

            else:
                return corrected

        attempt_prompt = f"""
    {prompt}

    RETRY FEEDBACK:
    {feedback}

    Return a newly revised script that resolves this feedback. Preserve all
    evidence-grounded corrections from the previous instructions.
    """.strip()

    raise ValueError(
        "Could not produce a valid corrected script after "
        f"{max_attempts} attempts. "
        + "; ".join(failures)
    )

def build_fact_check_report(
    claims: list[FactualClaim],
    evidence_bundles: list[ClaimEvidenceBundle],
    verifications: list[ClaimVerification],
    corrected_script: VideoScript,
) -> FactCheckReport:
    unresolved = [
        item for item in verifications
        if item.verdict == "insufficient_evidence"
        or (
            item.verdict in {"contradicted", "overstated"}
            and not item.correction
        )
    ]
    revised = any(
        item.verdict in {"contradicted", "overstated"}
        for item in verifications
    )

    if unresolved:
        verdict = "manual_review"
    elif revised:
        verdict = "revised"
    else:
        verdict = "pass"

    counts = {
        name: sum(item.verdict == name for item in verifications)
        for name in (
            "supported",
            "contradicted",
            "overstated",
            "insufficient_evidence",
        )
    }

    summary = (
        f"Checked {len(claims)} claims: "
        f"{counts['supported']} supported, "
        f"{counts['contradicted']} contradicted, "
        f"{counts['overstated']} overstated, and "
        f"{counts['insufficient_evidence']} unresolved."
    )

    return FactCheckReport(
        verdict=verdict,
        claims=claims,
        evidence_bundles=evidence_bundles,
        verifications=verifications,
        corrected_script=corrected_script,
        requires_manual_review=bool(unresolved),
        summary=summary,
    )


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_checked_script_filename(script: VideoScript) -> str:
    return f"{slugify(script.topic.title)}.json"


def build_report_filename(script: VideoScript) -> str:
    return f"{slugify(script.topic.title)}_fact_check.json"


def save_script(script: VideoScript, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(script.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_report(report: FactCheckReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def summarize_report(report: FactCheckReport) -> dict[str, object]:
    counts = {
        verdict: sum(
            item.verdict == verdict
            for item in report.verifications
        )
        for verdict in (
            "supported",
            "contradicted",
            "overstated",
            "insufficient_evidence",
        )
    }
    return {
        "overall_verdict": report.verdict,
        "claim_count": len(report.claims),
        **counts,
        "requires_manual_review": report.requires_manual_review,
        "corrected_word_count": report.corrected_script.word_count,
        "corrected_seconds": report.corrected_script.estimated_total_seconds,
    }

def _normalize_quote(text: str) -> str:
    """Normalize text before comparing evidence quotes."""
    text = text.lower()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s'-]", "", text)
    return text.strip()