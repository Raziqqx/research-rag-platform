"""PDF -> page-tagged plain text.

A document's canonical text is its cleaned pages joined by PAGE_SEP. Chunk offsets index into
that string, and `Document.page_of` maps any offset back to a 1-based page number, which is
what citations use. Parsing is deterministic, so offsets are reproducible across runs.
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

PAGE_SEP = "\n\n"

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SOFT_HYPHEN = re.compile(r"\u00ad\s*")
_WS = re.compile(r"[ \t\r\f\v]+")
_PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d{1,4}\s*$", re.IGNORECASE)


def clean_block(raw: str) -> str:
    """Join a layout block's lines into one paragraph: undo hyphenation, collapse spaces.

    NFKC folds typographic ligatures (e.g. U+FB03 "ffi") into plain letters so they match
    query terms; soft hyphens are line-break artefacts and are removed with their whitespace.
    """
    text = unicodedata.normalize("NFKC", raw.replace("\x00", ""))
    text = _SOFT_HYPHEN.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = text.replace("\n", " ")
    return _WS.sub(" ", text).strip()


def clean_page(blocks: list[str]) -> str:
    paras = [clean_block(b) for b in blocks]
    # Drop empty blocks and bare page-number footers; they only add noise to chunks.
    return "\n\n".join(p for p in paras if p and not _PAGE_NUMBER.match(p))


@dataclass
class Document:
    doc_id: str
    pages: list[str]
    text: str = field(init=False)
    page_starts: list[int] = field(init=False)

    def __post_init__(self) -> None:
        starts, pos = [], 0
        for p in self.pages:
            starts.append(pos)
            pos += len(p) + len(PAGE_SEP)
        self.page_starts = starts
        self.text = PAGE_SEP.join(self.pages)

    def page_of(self, offset: int) -> int:
        """1-based page containing character `offset` of `self.text`."""
        return max(1, bisect.bisect_right(self.page_starts, offset))


def parse_pdf(path: Path, doc_id: str) -> Document:
    pages = []
    with pymupdf.open(path) as pdf:
        for page in pdf:
            # "blocks" gives layout paragraphs in reading order; index 4 is the text and
            # index 6 the block type (0 = text, 1 = image).
            blocks = [b[4] for b in page.get_text("blocks", sort=True) if b[6] == 0]
            pages.append(clean_page(blocks))
    return Document(doc_id=doc_id, pages=pages)
