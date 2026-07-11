from __future__ import annotations

from abc import ABC, abstractmethod

from doc2knowledge.domain import Answer, RetrievedChunk


class GenerationService(ABC):
    @abstractmethod
    def generate(self, question: str, evidence: list[RetrievedChunk]) -> Answer:
        """Generate an answer grounded in the supplied evidence."""
