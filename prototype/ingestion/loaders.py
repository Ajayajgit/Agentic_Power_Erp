"""
ingestion/loaders.py

Small, single-purpose loader classes (Single Responsibility) - each knows
how to read exactly one kind of input file and nothing else. This keeps
IngestionPipeline free of file-format details.
"""

import json


class TranscriptLoader:
    @staticmethod
    def load(file_path: str) -> str:
        with open(file_path, "r") as f:
            return f.read()


class PlainTextLoader:
    """Used for the prior-engagement retrospective doc, same shape as TranscriptLoader
    but kept as a separate class since a real system would eventually parse
    richer document types (PDF/Word) here without touching transcript logic."""
    @staticmethod
    def load(file_path: str) -> str:
        with open(file_path, "r") as f:
            return f.read()


class TemplateLoader:
    @staticmethod
    def load(file_path: str) -> dict:
        with open(file_path, "r") as f:
            return json.load(f)
