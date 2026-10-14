"""Context assembly: bounded, labelled evidence taken only from one retrieval result.

Nothing here queries or re-ranks. Passages keep the order retrieval returned them. The
label-to-source map built here is what resolves citations later; it is never rebuilt from model text.
"""

from dataclasses import dataclass, field
from typing import Any

from building_with_rag.contracts import RetrievedChunk

MAX_PASSAGES = 8
MAX_CONTEXT_CHARS = 10_000


@dataclass(frozen=True)
class ContextPassage:
    label: str
    chunk: RetrievedChunk

    @property
    def section_label(self) -> str:
        """Act-qualified wording such as "BNS section 103", never a bare number."""
        label = (self.chunk.source or {}).get("section_label")
        if label:
            return str(label)
        prefix, _, number = self.chunk.section_id.partition(":")
        return f"{prefix.upper()} section {number}" if number else self.chunk.section_id

    def source_details(self) -> dict[str, Any]:
        source = self.chunk.source or {}
        return {
            key: source.get(key) for key in ("source_pdf", "page_start", "page_end", "source_id")
        }


@dataclass
class AssembledContext:
    passages: list[ContextPassage] = field(default_factory=list)
    outcome: str = "no_evidence"  # ok | truncated | no_evidence
    retrieved: int = 0
    chars: int = 0
    omitted: list[dict[str, str]] = field(default_factory=list)

    @property
    def by_label(self) -> dict[str, ContextPassage]:
        return {p.label: p for p in self.passages}

    def trace(self) -> dict[str, Any]:
        return {
            "retrieved": self.retrieved,
            "kept": len(self.passages),
            "chars": self.chars,
            "max_passages": MAX_PASSAGES,
            "max_chars": MAX_CONTEXT_CHARS,
            "omitted": self.omitted,
        }


def assemble(passages: list[RetrievedChunk]) -> AssembledContext:
    """Keep passages in retrieval order until the count or character budget is reached.

    A passage is never cut. The first one that does not fit, and every one after it, is left out.
    """
    context = AssembledContext(retrieved=len(passages))
    if not passages:
        return context

    stop_reason: str | None = None
    for chunk in passages:
        if stop_reason is None:
            if len(context.passages) >= MAX_PASSAGES:
                stop_reason = "passage_limit"
            elif context.chars + len(chunk.text) > MAX_CONTEXT_CHARS:
                stop_reason = "character_budget"
        if stop_reason is not None:
            context.omitted.append({"chunk_id": chunk.chunk_id, "reason": stop_reason})
            continue
        context.passages.append(ContextPassage(label=f"E{len(context.passages) + 1}", chunk=chunk))
        context.chars += len(chunk.text)

    context.outcome = "truncated" if context.omitted else "ok"
    return context
