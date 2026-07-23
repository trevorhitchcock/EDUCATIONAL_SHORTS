from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class KnowledgeNode(BaseModel):
    name: str = Field(
        description="The name of this knowledge category."
    )

    children: list["KnowledgeNode"] = Field(
        default_factory=list,
        description="The immediate subcategories of this category."
    )


class VideoTopic(BaseModel):
    title: str = Field(
        description="A concise and engaging title for the video."
    )

    category_path: list[str] = Field(
        description="The full knowledge-tree path associated with the topic."
    )

    learning_objective: str = Field(
        description="One sentence describing what the viewer should understand."
    )


class VideoTopicList(BaseModel):
    topics: list[VideoTopic] = Field(
        description="The generated video topics."
    )


class OutlineSection(BaseModel):
    section_type: str = Field(
        description=(
            "The role of this section, such as setup, explanation, example, "
            "comparison, or conclusion."
        )
    )

    purpose: str = Field(
        description="A short explanation of what this section accomplishes."
    )

    key_points: list[str] = Field(
        description="The essential facts or ideas that the script must cover."
    )

    visual_direction: str = Field(
        description="A concise suggestion for what viewers should see."
    )

    estimated_seconds: int = Field(
        ge=1,
        description="Estimated duration of this section in seconds."
    )


class VideoOutline(BaseModel):
    topic: VideoTopic = Field(
        description="The source topic used to create this outline."
    )

    hook: str = Field(
        description="The opening line or question designed to earn attention."
    )

    sections: list[OutlineSection] = Field(
        description="The ordered body sections of the video."
    )

    closing_takeaway: str = Field(
        description="The single idea the viewer should remember at the end."
    )

    estimated_total_seconds: int = Field(
        ge=1,
        description="Estimated duration of the complete video in seconds."
    )


class ScriptSegment(BaseModel):
    segment_type: str = Field(
        description=(
            "The role of this spoken segment, such as hook, explanation, "
            "example, significance, or closing."
        )
    )

    narration: str = Field(
        min_length=1,
        description="Natural spoken narration for this segment."
    )

    estimated_seconds: int = Field(
        ge=1,
        description="Estimated spoken duration of this segment in seconds."
    )


class VideoScript(BaseModel):
    topic: VideoTopic = Field(
        description="The source topic used to create the script."
    )

    hook: ScriptSegment = Field(
        description="The opening spoken segment."
    )

    sections: list[ScriptSegment] = Field(
        description="The ordered body narration segments."
    )

    closing: ScriptSegment = Field(
        description="The final spoken takeaway."
    )

    full_narration: str = Field(
        min_length=1,
        description="The complete narration in spoken order."
    )

    word_count: int = Field(
        ge=1,
        description="The number of words in the complete narration."
    )

    estimated_total_seconds: int = Field(
        ge=1,
        description="Estimated duration of the complete narration in seconds."
    )


class FactualClaim(BaseModel):
    claim_id: str = Field(
        min_length=1,
        description="A short unique identifier such as claim_1."
    )

    segment_type: str = Field(
        min_length=1,
        description="The script segment containing the claim."
    )

    original_text: str = Field(
        min_length=1,
        description="The exact sentence or wording from the script."
    )

    atomic_claim: str = Field(
        min_length=1,
        description="One independently verifiable factual proposition."
    )

    search_query: str = Field(
        min_length=1,
        description="A concise web search query for verifying the claim."
    )

    importance: Literal["low", "medium", "high"] = Field(
        description="How important it is to verify this claim."
    )


class ClaimList(BaseModel):
    claims: list[FactualClaim] = Field(
        default_factory=list,
        description="The most important factual claims requiring retrieval."
    )


class EvidenceSource(BaseModel):
    title: str = Field(
        min_length=1,
        description="The page or source title."
    )

    url: str = Field(
        min_length=1,
        description="The source URL."
    )

    domain: str = Field(
        min_length=1,
        description="The source domain."
    )

    excerpt: str = Field(
        min_length=1,
        description="A short passage relevant to the claim."
    )

    source_score: int = Field(
        description="A heuristic source-quality and relevance score."
    )

    retrieved_at: str = Field(
        min_length=1,
        description="UTC timestamp for retrieval."
    )


class ClaimEvidenceBundle(BaseModel):
    claim: FactualClaim = Field(
        description="The claim being checked."
    )

    sources: list[EvidenceSource] = Field(
        default_factory=list,
        description="Short retrieved passages relevant to the claim."
    )

    used_cache: bool = Field(
        default=False,
        description="Whether the evidence came from the local cache."
    )

    retrieval_error: str | None = Field(
        default=None,
        description="A retrieval error when no evidence could be collected."
    )


class ClaimVerification(BaseModel):
    claim_id: str = Field(
        min_length=1,
        description="The identifier of the verified claim."
    )

    verdict: Literal[
        "supported",
        "contradicted",
        "overstated",
        "insufficient_evidence",
    ] = Field(
        description="The claim's status according to the supplied evidence."
    )

    severity: Literal["none", "low", "medium", "high"] = Field(
        description="How seriously the original wording could mislead."
    )

    explanation: str = Field(
        min_length=1,
        description="A concise evidence-grounded explanation."
    )

    correction: str | None = Field(
        default=None,
        description="A concise replacement when the claim needs revision."
    )

    supporting_urls: list[str] = Field(
        default_factory=list,
        description="URLs that directly support the verdict."
    )

    evidence_quote: str | None = Field(
        default=None,
        description=(
            "An exact short passage from the supplied evidence that directly "
            "supports the verdict."
            ),
    )

    evidence_url: str | None = Field(
        default=None,
        description="The URL containing the evidence quote."
    )


class ClaimVerificationList(BaseModel):
    verifications: list[ClaimVerification] = Field(
        default_factory=list,
        description="One verification result for every supplied claim."
    )


class FactCheckReport(BaseModel):
    verdict: Literal["pass", "revised", "manual_review"] = Field(
        description="The overall result of the retrieval-grounded review."
    )

    claims: list[FactualClaim] = Field(
        default_factory=list,
        description="The extracted claims that were checked."
    )

    evidence_bundles: list[ClaimEvidenceBundle] = Field(
        default_factory=list,
        description="The retrieved evidence associated with each claim."
    )

    verifications: list[ClaimVerification] = Field(
        default_factory=list,
        description="Evidence-grounded verdicts for the claims."
    )

    corrected_script: VideoScript = Field(
        description="The script after applying evidence-grounded corrections."
    )

    requires_manual_review: bool = Field(
        description=(
            "Whether missing or conflicting evidence still requires a human."
        )
    )

    summary: str = Field(
        min_length=1,
        description="A concise summary of the fact-checking result."
    )


KnowledgeNode.model_rebuild()