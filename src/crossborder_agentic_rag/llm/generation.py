"""Minimal answer generation provider interface for Stage 6."""
from __future__ import annotations
from abc import ABC, abstractmethod

class BaseAnswerGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...

class TemplateAnswerGenerator(BaseAnswerGenerator):
    def generate(self, prompt: str) -> str: return prompt

def build_answer_generator(provider: str | None = None) -> BaseAnswerGenerator:
    name=(provider or "template").lower().replace("_","-")
    if name=="template": return TemplateAnswerGenerator()
    raise NotImplementedError(f"Answer generator provider '{name}' is not supported in Stage 6.")
