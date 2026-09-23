"""
ingestion/chunker.py

Simple word-based chunker with overlap. Mirrors the chunking strategy from
the design document (Section 1.3): transcripts are chunked on natural
breaks (blank lines / speaker turns) where possible, with overlap so a
requirement mentioned right at a chunk boundary isn't lost.
"""

from typing import List


class TextChunker:
    def __init__(self, chunk_size_words: int = 180, overlap_words: int = 30):
        self._chunk_size = chunk_size_words
        self._overlap = overlap_words

    def chunk(self, text: str) -> List[str]:
        """
        Chunk on paragraph/speaker-turn boundaries first (splitting on blank
        lines), then greedily pack those units up to chunk_size_words, with
        a sliding overlap between chunks.
        """
        units = [u.strip() for u in text.split("\n\n") if u.strip()]
        if not units:
            return []

        chunks: List[str] = []
        current_words: List[str] = []

        for unit in units:
            unit_words = unit.split()
            if len(current_words) + len(unit_words) > self._chunk_size and current_words:
                chunks.append(" ".join(current_words))
                # start next chunk with overlap from the tail of the previous one
                current_words = current_words[-self._overlap:] + unit_words
            else:
                current_words.extend(unit_words)

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks
