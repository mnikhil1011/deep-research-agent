"""Plain-code text helpers shared by the app and the evaluation harness."""

import re

# Sentence-ending punctuation, any closing quotes or brackets, and any citation
# markers such as [3] or [@key] that trail it, followed by a new sentence.
_BOUNDARY = re.compile(r"""([.!?]+["')\]*_]*(?:\s*\[[^\]\s]+\])*)\s+(?=["'(\[*_]*[A-Z0-9])""")
_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "dr", "mr", "mrs", "ms", "prof", "inc", "ltd",
    "co", "corp", "fig", "approx", "st", "jr", "sr", "al", "eq", "sec", "cf",
}  # fmt: skip
_DOTTED = re.compile(r"(?:[A-Za-z]\.)*[A-Za-z]")  # "U.S", "e.g", or a single initial


def _is_abbreviation(text_before: str) -> bool:
    words = text_before.split()
    if not words:
        return False
    word = words[-1].lstrip("\"'([*_")
    return word.lower() in _ABBREVIATIONS or bool(_DOTTED.fullmatch(word))


def split_sentences(paragraph: str) -> list[str]:
    """Split one paragraph into sentences, keeping trailing citation markers attached."""
    pieces, start = [], 0
    for match in _BOUNDARY.finditer(paragraph):
        punctuation = match.group(1)
        if punctuation.startswith(".") and not punctuation.startswith(".."):
            if _is_abbreviation(paragraph[start : match.start(1)]):
                continue
        pieces.append(paragraph[start : match.end(1)])
        start = match.end()
    pieces.append(paragraph[start:])
    return [piece for piece in pieces if piece.strip()]


def normalize(text: str) -> str:
    """Lower-case and reduce to letters, digits and single spaces, for fuzzy comparison."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
