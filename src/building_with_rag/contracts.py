"""Shared typed contracts. Later stories extend these additively."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Pattern = Literal["semantic", "hybrid", "hybrid-reranked", "structured", "decomposition", "hyde"]


# A filter value is a short plain token: no operators, no "$", no MongoDB fragments.
FilterValue = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]
FilterList = Annotated[list[FilterValue], Field(min_length=1, max_length=10)]


class SemanticFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    act: FilterList | None = None
    status: FilterList | None = None
    access_level: FilterList | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    pattern: Pattern
    caller_id: str | None = None
    filters: SemanticFilters | None = None
    limit: int = Field(default=5, ge=1, le=20)
    generate_answer: bool = False
    required_acts: list[str] | None = None
    chapter: str | None = None

    @field_validator("question", mode="before")
    @classmethod
    def _strip_question(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class RetrievedChunk(BaseModel):
    chunk_id: str
    section_id: str
    act: str
    text: str
    heading: str | None = None
    score: float | None = None
    source: dict[str, Any] | None = None


class OmittedCandidate(BaseModel):
    chunk_id: str
    omitted_reason: str


class SubquestionEvidence(BaseModel):
    subquestion: str
    status: Literal["evidenced", "no_evidence"]
    results: list[RetrievedChunk] = []
    reason: str | None = None


class StructuredSignals(BaseModel):
    intent: Literal["exact_lookup", "filter", "aggregation"]
    act: str | None = None
    section_number: str | None = None
    chapter: str | None = None


class GenerationResult(BaseModel):
    outcome: Literal["answered", "insufficient_evidence", "unavailable", "malformed"]
    answer: str | None = None
    claims: list[Any] = []
    citations: list[Any] = []
    supporting_passages: list[RetrievedChunk] = []
    provider: str | None = None
    model: str | None = None
    trace: dict[str, Any] = {}
    context_outcome: str | None = None
    confidence: float | None = None
    draft_answer: str | None = None
    issues: list[str] = []
    attempts: int | None = None
    low_confidence_reason: str | None = None


class QueryResult(BaseModel):
    pattern: str
    status: str
    message: str
    trace: dict[str, Any] = {}
    results: list[RetrievedChunk] = []
    generation: GenerationResult | None = None
    omitted_candidates: list[OmittedCandidate] = []
    subquestions: list[SubquestionEvidence] = []
    hyde_direct_candidates: list[RetrievedChunk] = []
    hyde_query_candidates: list[RetrievedChunk] = []
    hyde_hypothetical_text_debug: str | None = None


# --- OpenAI-compatible contracts (text only) ---


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str


class RagOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: Pattern | None = None
    filters: SemanticFilters | None = None
    act: FilterList | None = None
    status: FilterList | None = None
    access_level: FilterList | None = None
    limit: int | None = Field(default=None, ge=1, le=20)
    required_acts: list[str] | None = None
    chapter: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    n: int = Field(default=1, ge=1, le=1)
    rag_options: RagOptions | None = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "building-with-rag"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class OpenAIErrorBody(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class OpenAIError(BaseModel):
    error: OpenAIErrorBody


# --- MongoDB schema contracts (document shapes only; no database yet) ---


class ChunkDocument(BaseModel):
    chunk_id: str
    section_id: str
    act: str
    heading: str | None = None
    text: str
    source: dict[str, Any] | None = None
    embedding: list[float] | None = None  # voyage-3.5, 1,024 dimensions
