"""Rule-based classifier for the structured mode: question text -> StructuredSignals.

Pure and offline: fixed regular expressions only, no model, network or database. It is the only
reader of the question text in the structured mode. It never guesses: anything that is not exactly
one BNS or IPC act plus one plain integer section is reported with a reason instead.
"""

import re
from dataclasses import dataclass

from building_with_rag.contracts import StructuredSignals

CLASSIFIER_NAME = "structured-rules-v1"
MIN_SECTION = 1
MAX_SECTION = 999

_BNS = re.compile(r"\bBNS\b|\bBharatiya\s+Nyaya\s+Sanhita\b", re.IGNORECASE)
_IPC = re.compile(r"\bIPC\b|\bIndian\s+Penal\s+Code\b", re.IGNORECASE)

_NUM = r"\d+[A-Za-z]?"
_SEP = r"(?:\s*(?:,|and|to|-|–|&)\s*)"
# "section 103", "sections 100 to 105", "sec. 302", "s. 302"
_AFTER_KEYWORD = re.compile(rf"\b(?:sections?|secs?\b\.?|s\.)\s*({_NUM}(?:{_SEP}{_NUM})*)", re.IGNORECASE)
# "BNS 103", "IPC 302"
_AFTER_ACT = re.compile(rf"\b(?:BNS|IPC)\s+({_NUM})\b", re.IGNORECASE)
_SECTION_WORD = re.compile(r"\bsections?\b", re.IGNORECASE)
_TOKEN = re.compile(_NUM)

_AGGREGATION = re.compile(
    r"\bhow\s+many\s+sections\b|\bnumber\s+of\s+sections\b|\bcount\s+(?:of\s+)?(?:the\s+)?sections\b",
    re.IGNORECASE,
)
_FILTER = re.compile(r"\b(?:list|show)\b.*\bsections\b.*\bchapter\b", re.IGNORECASE)
_CHAPTER = re.compile(r"^[IVXL]{1,6}$")


@dataclass(frozen=True)
class Classification:
    """`kind` is exact, clarification or recommendation; `signals` is set for exact and for a
    recognised filter or aggregation phrase."""

    kind: str
    reason: str | None
    rule: str
    signals: StructuredSignals | None = None


def validate_chapter(chapter: str | None) -> str | None:
    """Roman numerals as stored (I to XXIII), upper-cased. None or an invalid value gives None."""
    if chapter is None:
        return None
    value = chapter.strip().upper()
    return value if _CHAPTER.match(value) else None


def _act_code(question: str) -> tuple[str | None, bool]:
    bns, ipc = bool(_BNS.search(question)), bool(_IPC.search(question))
    if bns and ipc:
        return None, True
    return ("BNS" if bns else "IPC" if ipc else None), False


def _section_tokens(question: str) -> list[str]:
    tokens: list[str] = []
    for match in _AFTER_KEYWORD.finditer(question):
        tokens.extend(_TOKEN.findall(match.group(1)))
    for match in _AFTER_ACT.finditer(question):
        tokens.append(match.group(1))
    return tokens


def classify(question: str, chapter: str | None = None) -> Classification:
    question = question.strip()
    if chapter is not None and validate_chapter(chapter) is None:
        return Classification("clarification", "invalid_chapter", "chapter_validation")
    chapter = validate_chapter(chapter)
    act, both = _act_code(question)

    if _AGGREGATION.search(question):
        signals = StructuredSignals(intent="aggregation", act=act, chapter=chapter)
        return Classification("recommendation", "intent_not_executed", "aggregation_phrase", signals)
    if _FILTER.search(question):
        signals = StructuredSignals(intent="filter", act=act, chapter=chapter)
        return Classification("recommendation", "intent_not_executed", "filter_phrase", signals)

    if both:
        return Classification("clarification", "ambiguous_act", "both_acts")
    tokens = _section_tokens(question)
    if not tokens:
        if act and _SECTION_WORD.search(question):
            return Classification("clarification", "missing_section", "act_without_section")
        return Classification("recommendation", "not_structured_request", "no_structured_phrase")
    if len({t.upper() for t in tokens}) > 1:
        return Classification("clarification", "multiple_sections", "several_sections")
    token = tokens[0]
    if not token.isdigit():
        return Classification("clarification", "lettered_section_unsupported", "lettered_section")
    if not MIN_SECTION <= int(token) <= MAX_SECTION:
        return Classification("clarification", "section_out_of_range", "section_range_check")
    if act is None:
        return Classification("clarification", "missing_act", "section_without_act")
    signals = StructuredSignals(
        intent="exact_lookup", act=act, section_number=str(int(token)), chapter=chapter
    )
    return Classification("exact", None, "act_and_section", signals)
