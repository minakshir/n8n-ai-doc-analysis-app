"""Split extracted text into retrieval chunks.

Chunks never cross a PDF page, so every retrieved passage cites one page.
Paragraphs are packed up to TARGET characters; a paragraph that is too long is
split at sentence ends, then at word boundaries. Consecutive chunks share up to
OVERLAP characters of whole paragraphs/sentences so an answer spanning a
boundary is still retrievable.
"""

import re
from dataclasses import dataclass

TARGET = 1200
OVERLAP = 200
_PAGE = re.compile(r"^\[Page (\d+)\]\n?", re.M)
_PARA = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    page: int | None
    text: str


def chunk_text(text: str, kind: str) -> list[Chunk]:
    chunks = []
    for page, body in _pages(text, kind):
        units = [u for p in _PARA.split(body) for u in _fit(p.strip()) if u]
        buf: list[str] = []
        for unit in units:
            if buf and sum(len(u) + 1 for u in buf) + len(unit) > TARGET:
                chunks.append(Chunk(page, "\n".join(buf)))
                buf = _tail(buf)
                if sum(len(u) + 1 for u in buf) + len(unit) > TARGET:
                    buf = []
            buf.append(unit)
        if buf:
            chunks.append(Chunk(page, "\n".join(buf)))
    return chunks


def _pages(text: str, kind: str) -> list[tuple[int | None, str]]:
    if kind != "pdf":
        return [(None, text)]
    parts = _PAGE.split(text)  # ['', '1', body1, '2', body2, ...]
    return [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def _fit(paragraph: str) -> list[str]:
    if len(paragraph) <= TARGET:
        return [paragraph]
    out = []
    for sentence in _SENTENCE.split(paragraph):
        if len(sentence) <= TARGET:
            out.append(sentence)
            continue
        piece = ""
        for word in sentence.split():
            while len(word) > TARGET:  # no whitespace to split on
                if piece:
                    out.append(piece)
                    piece = ""
                out.append(word[:TARGET])
                word = word[TARGET:]
            if piece and len(piece) + 1 + len(word) > TARGET:
                out.append(piece)
                piece = word
            else:
                piece = f"{piece} {word}" if piece else word
        if piece:
            out.append(piece)
    return out


def _tail(units: list[str]) -> list[str]:
    tail, size = [], 0
    for unit in reversed(units[1:]):  # never repeat the whole previous chunk
        size += len(unit) + 1
        if size > OVERLAP:
            break
        tail.insert(0, unit)
    return tail
