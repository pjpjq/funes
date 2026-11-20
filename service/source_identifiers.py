"""Frozen logical value for PostgreSQL derived-v1 search identifiers."""
import re
from typing import Any


_V1_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,63}")


def technical_index_text_v1(raw_text: Any, retrieval_text: Any) -> str:
    terms = []
    seen = set()
    for value in (raw_text, retrieval_text):
        for term in _V1_TERM_RE.findall(str(value or "")):
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            terms.append(term)
            if len(terms) >= 256:
                return " ".join(terms)
    return " ".join(terms)


def logical_search_identifiers(
    stored: str | None, raw_text: str, retrieval_text: str | None,
) -> str:
    """Decode an already text-decoded value under a validated source schema."""
    if stored is not None:
        return stored
    return technical_index_text_v1(raw_text, retrieval_text)
