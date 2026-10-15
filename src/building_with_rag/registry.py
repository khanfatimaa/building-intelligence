"""The one shared registry of course modes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Mode:
    pattern: str
    model_id: str


MODES: tuple[Mode, ...] = tuple(
    Mode(p, f"rag-{p}")
    for p in ("semantic", "hybrid", "hybrid-reranked", "structured", "decomposition", "hyde", "agentic")
)
PATTERNS = tuple(m.pattern for m in MODES)
BY_MODEL_ID = {m.model_id: m for m in MODES}
