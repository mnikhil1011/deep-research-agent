"""Reads a LaTeX paper into sentences, citations and a reference list.

    load_paper("draft/main.tex")      a file
    load_paper("draft/")              a folder; the main file is found
    load_paper("arxiv:1706.03762")    the source is downloaded from arXiv

This is plain code. Which sentences cite which references has to be exact and
repeatable, so no model is involved.
"""

import gzip
import io
import re
import tarfile
from pathlib import Path

from pylatexenc import latex2text, latexwalker, macrospec

from app import config
from app.audit.models import CitedSentence, Paper, Reference
from app.text import split_sentences
from app.tools import scholar

OPENING_CHARS = 6000
_CITE_MACROS = [
    "cite", "citep", "citet", "citealp", "citealt", "citeauthor", "citeyear", "citeyearpar",
    "citenum", "parencite", "textcite", "autocite", "footcite", "smartcite", "supercite",
    "Cite", "Citep", "Citet", "Citealp", "Citealt", "Parencite", "Textcite", "Autocite",
]  # fmt: skip
_REF_MACROS = ["ref", "eqref", "cref", "Cref", "autoref", "pageref", "nameref"]
# Floats, maths and listings are dropped: the audit is about prose that cites.
_DROPPED_ENVIRONMENTS = [
    "figure", "figure*", "table", "table*", "tabular", "tabular*", "equation", "equation*",
    "align", "align*", "gather", "gather*", "multline", "multline*", "eqnarray", "eqnarray*",
    "displaymath", "algorithm", "algorithm*", "algorithmic", "lstlisting", "verbatim", "minted",
    "tikzpicture", "thebibliography", "comment", "tabularx", "longtable", "wrapfigure",
    "wraptable", "sidewaystable", "subfigure", "subtable",
]  # fmt: skip
# Macros whose content is not prose: notes, labels, links and the title block.
_SILENT_MACROS = [
    "footnote", "nocite", "label", "url", "maketitle", "title", "author", "date", "thanks",
    "includegraphics", "vspace", "hspace",
]  # fmt: skip
_SECTION = "@@SECTION "
_MARKER = re.compile(r"\[@([^\]]+)\]")
_NEW_ARXIV = re.compile(r"\d{4}\.\d{4,5}")
# Matches arXiv:1706.03762, arxiv.org/abs/1706.03762 and the older "CoRR, abs/1409.0473".
_ARXIV_IN_TEXT = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arxiv[:\s]\s*|\babs/)(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})",
    re.IGNORECASE,
)
# A fragment counts as a sentence if it ends like one, or is long enough to be
# one anyway. This keeps "We use Adam [k]." and drops table rows such as
# "ConvS2S [k] 25.16 40.46".
MIN_SENTENCE_WORDS = 3
LONG_ENOUGH_WORDS = 10
# A DOI after "doi.org/" or "doi:", or standing alone. A "10.xxxx/" buried in
# some other URL's path is not taken, because it usually carries extra path parts.
_DOI_IN_TEXT = re.compile(
    r"(?:doi\.org/|doi:\s*|(?<![\w/.]))(10\.\d{4,9}/[^\s{}\\\"<>]+)", re.IGNORECASE
)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_CITE_IN_SOURCE = re.compile(
    r"\\(?:[A-Za-z]*[Cc]ite[A-Za-z]*)\*?(?:\s*\[[^\]]*\])*\s*\{([^}]+)\}"
)
_BIBTEX_FIELD = re.compile(r"([A-Za-z][\w-]*)\s*=\s*")


class PaperError(ValueError):
    """The paper could not be read."""


# --- LaTeX to text -----------------------------------------------------------


def _last_argument(node, l2tobj) -> str:
    """The text of a macro's final argument, or "" when the parser found none."""
    arguments = getattr(node.nodeargd, "argnlist", None) or []
    if not arguments or arguments[-1] is None:
        return ""
    return l2tobj.nodelist_to_text([arguments[-1]])


def _cite_marker(node, l2tobj) -> str:
    cleaned = [key.strip() for key in _last_argument(node, l2tobj).split(",") if key.strip()]
    return f" [@{','.join(cleaned)}]" if cleaned else ""


def _section_marker(node, l2tobj) -> str:
    return f"\n\n{_SECTION}{' '.join(_last_argument(node, l2tobj).split())}\n\n"


def _converter() -> latex2text.LatexNodes2Text:
    text_db = latex2text.get_default_latex_context_db()
    text_db.add_context_category(
        "paper-audit",
        prepend=True,
        macros=[latex2text.MacroTextSpec(name, simplify_repl=_cite_marker) for name in _CITE_MACROS]
        + [latex2text.MacroTextSpec(name, simplify_repl="[ref]") for name in _REF_MACROS]
        + [latex2text.MacroTextSpec(name, simplify_repl="") for name in _SILENT_MACROS]
        + [latex2text.MacroTextSpec("href", simplify_repl=_last_argument)]
        + [latex2text.MacroTextSpec("section", simplify_repl=_section_marker)]
        + [
            latex2text.MacroTextSpec(name, simplify_repl="\n\n")
            for name in ("part", "chapter", "subsection", "subsubsection", "paragraph")
        ],
        environments=[
            latex2text.EnvironmentTextSpec(name, simplify_repl="") for name in _DROPPED_ENVIRONMENTS
        ],
    )
    return latex2text.LatexNodes2Text(latex_context=text_db, math_mode="verbatim")


def _walker_context():
    walker_db = latexwalker.get_default_latex_context_db()
    walker_db.add_context_category(
        "paper-audit",
        prepend=True,
        macros=[macrospec.MacroSpec(name, "*[[{") for name in _CITE_MACROS + ["nocite"]]
        + [macrospec.MacroSpec(name, "*{") for name in _REF_MACROS]
        + [macrospec.MacroSpec(name, "*[{") for name in ("paragraph", "subsubsection")]
        + [macrospec.MacroSpec("href", "{{"), macrospec.MacroSpec("url", "{")],
    )
    return walker_db


def to_text(latex: str) -> str:
    """Convert LaTeX to plain text. Citations become [@key1,key2] markers."""
    walker = latexwalker.LatexWalker(latex, latex_context=_walker_context(), tolerant_parsing=True)
    nodes, _, _ = walker.get_latex_nodes()
    return _converter().nodelist_to_text(nodes)


def _plain(latex: str) -> str:
    """LaTeX to one line of plain text, with no citation markers."""
    return " ".join(_MARKER.sub("", to_text(latex)).split())


# --- Reading the source ------------------------------------------------------


def _read(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def _inline_inputs(path: Path, root: Path, depth: int = 0) -> str:
    """Return a file's text with every \\input and \\include replaced by its contents."""
    text = _strip_comments(_read(path))
    if depth >= 10:
        return text

    def replace(match: re.Match) -> str:
        name = match.group(1).strip()
        for candidate in (path.parent / name, root / name):
            candidate = candidate if candidate.suffix else candidate.with_suffix(".tex")
            # Only files inside the paper's own folder are read.
            if candidate.is_file() and root.resolve() in candidate.resolve().parents:
                return _inline_inputs(candidate, root, depth + 1)
        return ""

    return re.sub(r"\\(?:input|include|subfile)\s*\{([^}]+)\}", replace, text)


def _braced(text: str, start: int) -> str:
    """The contents of the first balanced {...} at or after `start`."""
    open_at = text.find("{", start)
    if open_at < 0:
        return ""
    depth, position = 0, open_at
    while position < len(text):
        char = text[position]
        if char == "\\":
            position += 2  # an escaped character, such as \{ or \}
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1 : position]
        position += 1
    return ""


def find_main_tex(directory: Path) -> Path:
    """Pick the file that holds \\documentclass and \\begin{document}."""
    candidates = []
    for path in sorted(directory.rglob("*.tex")):
        text = _strip_comments(_read(path))
        if "\\documentclass" in text:
            candidates.append(("\\begin{document}" in text, len(text), path))
    if not candidates:
        raise PaperError(f"No .tex file with \\documentclass was found in {directory}")
    return max(candidates)[2]


def fetch_arxiv_source(arxiv_id: str) -> Path:
    """Download and unpack a paper's LaTeX source from arXiv. Returns its folder."""
    folder = Path(config.CACHE_DIR) / "arxiv" / arxiv_id.replace("/", "_")
    if folder.is_dir() and any(folder.rglob("*.tex")):
        return folder
    content = scholar.download(f"https://arxiv.org/e-print/{arxiv_id}")
    if content.startswith(b"%PDF"):
        raise PaperError(f"arXiv has no LaTeX source for {arxiv_id}, only a PDF")
    if content.startswith(b"\x1f\x8b"):
        content = gzip.decompress(content)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(content)) as archive:
            archive.extractall(folder, filter="data")  # "data" refuses unsafe paths and links
    except tarfile.ReadError:
        (folder / "main.tex").write_bytes(content)  # a single TeX file, not an archive
    return folder


# --- Bibliography ------------------------------------------------------------


def _identifiers(text: str) -> tuple[str, str]:
    """Find a DOI and an arXiv id anywhere in a reference's text."""
    doi = _DOI_IN_TEXT.search(text)
    arxiv = _ARXIV_IN_TEXT.search(text)
    return (
        doi.group(1).rstrip(".,;)").lower() if doi else "",
        arxiv.group(1) if arxiv else "",
    )


def _bibtex_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    position = 0
    while match := _BIBTEX_FIELD.search(body, position):
        name, start = match.group(1).lower(), match.end()
        if body[start : start + 1] == "{":
            value = _braced(body, start)
            position = start + len(value) + 2
        elif body[start : start + 1] == '"':
            end = start + 1
            depth = 0
            while end < len(body) and (body[end] != '"' or depth):
                depth += {"{": 1, "}": -1}.get(body[end], 0)
                end += 1
            value = body[start + 1 : end]
            position = end + 1
        else:
            end = body.find(",", start)
            end = len(body) if end < 0 else end
            value = body[start:end].strip()
            position = end
        fields[name] = value
    return fields


def _author_names(field: str) -> list[str]:
    names = []
    for name in re.split(r"\s+and\s+", _plain(field)):
        if "," in name:  # "Last, First" becomes "First Last"
            last, _, first = name.partition(",")
            name = f"{first.strip()} {last.strip()}"
        if name.strip() and name.strip().lower() != "others":
            names.append(" ".join(name.split()))
    return names


def parse_bibtex(text: str) -> dict[str, Reference]:
    """Parse a .bib file into references, keyed by citation key."""
    references: dict[str, Reference] = {}
    for match in re.finditer(r"@(\w+)\s*\{\s*([^,\s{}]+)\s*,", text):
        if match.group(1).lower() in ("comment", "string", "preamble"):
            continue
        body = _braced(text, match.start())
        fields = _bibtex_fields(body[body.find(",") + 1 :])
        doi, arxiv_id = _identifiers(" ".join(fields.values()))
        eprint = fields.get("eprint", "").strip()
        if not arxiv_id and _NEW_ARXIV.fullmatch(eprint):
            arxiv_id = eprint
        year = _YEAR.search(fields.get("year", "") or fields.get("date", ""))
        title = _plain(fields.get("title", ""))
        authors = _author_names(fields.get("author", ""))
        venue = _plain(fields.get("journal", "") or fields.get("booktitle", ""))
        raw = ". ".join(part for part in (", ".join(authors), title, venue, year and year.group()) if part)
        references[match.group(2)] = Reference(
            key=match.group(2),
            raw=raw,
            title=title,
            authors=authors,
            year=int(year.group()) if year else None,
            doi=fields.get("doi", "").strip().lower().removeprefix("https://doi.org/") or doi,
            arxiv_id=arxiv_id,
        )
    return references


def parse_bbl(text: str) -> dict[str, Reference]:
    """Parse a BibTeX-generated bibliography (\\bibitem entries) into references."""
    references: dict[str, Reference] = {}
    starts = [m.start() for m in re.finditer(r"\\bibitem\b", text)]
    end_of_list = text.find("\\end{thebibliography}")
    end_of_list = len(text) if end_of_list < 0 else end_of_list
    for index, start in enumerate(starts):
        entry = text[start : starts[index + 1] if index + 1 < len(starts) else end_of_list]
        position = len("\\bibitem")
        rest = entry[position:].lstrip()
        if rest.startswith("["):  # an optional label such as [Vaswani et al.(2017)]
            depth = 0
            for offset, char in enumerate(rest):
                depth += {"{": 1, "}": -1}.get(char, 0)
                if char == "]" and depth == 0:
                    rest = rest[offset + 1 :]
                    break
        key = _braced(rest, 0).strip()
        body = rest[rest.find("}") + 1 :] if key else ""
        if not key or not body.strip():
            continue
        blocks = [_plain(block) for block in body.split("\\newblock")]
        raw = " ".join(block for block in blocks if block)
        doi, arxiv_id = _identifiers(body)
        years = _YEAR.findall(raw)
        # In the common styles the blocks run: authors, title, venue.
        title = blocks[1].rstrip(".") if len(blocks) >= 3 else ""
        references[key] = Reference(
            key=key,
            raw=raw,
            title=title,
            authors=[blocks[0].rstrip(".")] if len(blocks) >= 3 and blocks[0] else [],
            year=int(years[-1]) if years else None,
            doi=doi,
            arxiv_id=arxiv_id,
        )
    return references


def _bibliography(
    document: str, main_tex: Path, root: Path
) -> tuple[dict[str, Reference], dict[str, Reference]]:
    """Return (printed, database).

    `printed` is the reference list as it appears in the paper, from a compiled
    .bbl or one typed in by hand. `database` is everything in the .bib files,
    which usually holds many entries the paper never uses.
    """
    printed: dict[str, Reference] = {}
    bbl_files = [main_tex.with_suffix(".bbl"), *sorted(root.glob("*.bbl"))]
    for path in dict.fromkeys(p for p in bbl_files if p.is_file()):
        printed |= parse_bbl(_read(path))
    printed |= parse_bbl(document)
    database: dict[str, Reference] = {}
    names = re.findall(r"\\(?:bibliography|addbibresource)\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", document)
    for name in (part.strip() for group in names for part in group.split(",")):
        for candidate in (main_tex.parent / name, root / name):
            candidate = candidate if candidate.suffix == ".bib" else candidate.with_suffix(".bib")
            if candidate.is_file():
                database |= parse_bibtex(_read(candidate))
                break
    return printed, database


# --- The paper ---------------------------------------------------------------


def _cited_sentences(body_text: str) -> list[CitedSentence]:
    sentences: list[CitedSentence] = []
    section = ""
    for paragraph in re.split(r"\n\s*\n", body_text):
        paragraph = " ".join(paragraph.split())
        if paragraph.startswith(_SECTION.strip()):
            section = paragraph[len(_SECTION.strip()) :].strip()
            continue
        previous = ""
        for sentence in split_sentences(paragraph):
            keys = list(
                dict.fromkeys(key for group in _MARKER.findall(sentence) for key in group.split(","))
            )
            readable = _MARKER.sub(lambda m: "[" + ", ".join(m.group(1).split(",")) + "]", sentence)
            readable = re.sub(r"\s+([.,;:!?])", r"\1", readable).strip()
            words = re.findall(r"[A-Za-z]{2,}", _MARKER.sub("", sentence))
            ends_like_a_sentence = readable.rstrip("\"')]*_ ").endswith((".", "!", "?", ":", ";"))
            is_prose = len(words) >= LONG_ENOUGH_WORDS or (
                len(words) >= MIN_SENTENCE_WORDS and ends_like_a_sentence
            )
            if keys and is_prose:
                sentences.append(
                    CitedSentence(
                        id=len(sentences), text=readable, context=previous, section=section, keys=keys
                    )
                )
            previous = readable
    return sentences


def parse_paper(main_tex: Path) -> Paper:
    """Parse a LaTeX paper, given its main file."""
    root = main_tex.parent
    document = _inline_inputs(main_tex, root)
    if "\\begin{document}" not in document:
        raise PaperError(f"{main_tex} has no \\begin{{document}}")
    preamble, _, body = document.partition("\\begin{document}")
    body = body.partition("\\end{document}")[0]

    title_at = re.search(r"\\(?:icml)?title\s*(?:\[[^\]]*\])?\s*\{", document)
    title = _plain(_braced(document, title_at.start())) if title_at else ""
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", document, re.DOTALL)
    abstract = _plain(abstract_match.group(1)) if abstract_match else ""
    if abstract_match:
        body = body.replace(abstract_match.group(0), "")

    body_text = to_text(body)
    sentences = _cited_sentences(body_text)
    printed, database = _bibliography(document, main_tex, root)
    # Citations inside tables and captions count as citations, though the text
    # around them is not audited, so they are collected from the raw source.
    cited = list(
        dict.fromkeys(
            key.strip()
            for group in _CITE_IN_SOURCE.findall(body)
            for key in group.split(",")
            if key.strip()
        )
    )
    # A .bib entry is structured, so it is preferred over the same key's printed form.
    references = printed | {key: database[key] for key in cited if key in database}
    opening = " ".join(_MARKER.sub("", body_text.replace(_SECTION.strip(), "")).split())
    return Paper(
        title=title,
        abstract=abstract,
        opening=opening[:OPENING_CHARS],
        sentences=sentences,
        references=references,
        missing_keys=[key for key in cited if key not in references],
        uncited_keys=[key for key in printed if key not in cited],
    )


def load_paper(source: str) -> Paper:
    """Load a paper from a .tex file, a folder, or an arXiv id such as arxiv:1706.03762."""
    arxiv = re.fullmatch(r"(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)", source.strip(), re.IGNORECASE)
    if arxiv:
        return parse_paper(find_main_tex(fetch_arxiv_source(arxiv.group(1))))
    path = Path(source)
    if path.is_dir():
        return parse_paper(find_main_tex(path))
    if path.is_file():
        return parse_paper(path)
    raise PaperError(f"{source!r} is not a file, a folder or an arXiv id")
