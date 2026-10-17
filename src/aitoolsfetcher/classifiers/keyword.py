"""Keyword-based AI tool classifier — zero external dependencies."""

from __future__ import annotations

import re
from collections import defaultdict

from aitoolsfetcher.classifiers.base import BaseClassifier, ClassificationResult
from aitoolsfetcher.core.models import AICategory, AITool

# Curated keyword taxonomy with weights
# Higher weight = stronger signal of that category
_TAXONOMY: dict[AICategory, dict[str, float]] = {
    AICategory.AGENT: {
        "agent": 3.0, "autonomous": 2.5, "multi-agent": 3.0, "ai-agent": 3.0,
        "tool-use": 2.0, "function-calling": 2.0, "mcp": 2.5, "a2a": 2.5,
        "agentic": 3.0, "orchestrat": 2.0, "autogen": 3.0, "crewai": 3.0,
        "langraph": 3.0, "swarm": 2.0, "planning": 1.5, "reasoning": 1.5,
        "react": 1.0, "chain-of-thought": 2.0, "self-improving": 2.0,
    },
    AICategory.FRAMEWORK: {
        "framework": 2.5, "langchain": 3.0, "llamaindex": 3.0, "haystack": 3.0,
        "pytorch": 3.0, "tensorflow": 3.0, "jax": 2.5, "keras": 3.0,
        "huggingface": 2.0, "transformers": 2.5, "pipeline": 1.5,
        "deep-learning": 2.0, "machine-learning": 2.0, "neural": 1.5,
        "training": 1.5, "dspy": 3.0, "semantic-kernel": 3.0,
    },
    AICategory.MODEL: {
        "model": 1.5, "llm": 3.0, "large-language-model": 3.0, "gpt": 2.5,
        "claude": 2.0, "llama": 2.5, "mistral": 2.5, "gemma": 2.5,
        "diffusion": 2.5, "stable-diffusion": 3.0, "transformer": 2.0,
        "whisper": 2.5, "embedding": 2.0, "foundation-model": 3.0,
        "pretrained": 2.0, "checkpoint": 1.5, "weights": 1.5,
        "text-generation": 2.0, "image-generation": 2.0, "speech": 1.5,
    },
    AICategory.DATASET: {
        "dataset": 3.0, "benchmark": 2.5, "corpus": 2.5, "annotation": 2.0,
        "labeled": 2.0, "training-data": 3.0, "evaluation-data": 2.5,
        "synthetic-data": 2.5, "data-generation": 2.0,
    },
    AICategory.INFRA: {
        "vector-database": 3.0, "vector-db": 3.0, "vectorstore": 3.0,
        "rag": 3.0, "retrieval": 2.0, "inference": 2.5, "serving": 2.5,
        "deployment": 2.0, "fine-tuning": 3.0, "quantization": 3.0,
        "gguf": 3.0, "ggml": 3.0, "vllm": 3.0, "ollama": 3.0,
        "mlops": 2.5, "gpu": 1.5, "distributed": 1.5, "scaling": 1.5,
        "lora": 3.0, "peft": 3.0, "qlora": 3.0, "pruning": 2.0,
        "knowledge-base": 2.0, "embedding-store": 2.5,
    },
    AICategory.DEV_TOOL: {
        "prompt-engineering": 3.0, "prompt": 1.5, "evaluation": 2.0,
        "eval": 1.5, "testing": 1.0, "monitoring": 1.5, "observability": 1.5,
        "playground": 2.0, "debugg": 1.5, "tracing": 1.5, "logging": 1.0,
        "guardrail": 2.5, "safety": 1.5, "alignment": 2.0, "red-team": 2.0,
        "explainab": 2.0, "interpretab": 2.0,
    },
    AICategory.LIBRARY: {
        "library": 2.0, "sdk": 2.5, "api-client": 2.0, "wrapper": 1.5,
        "package": 1.0, "binding": 1.5, "openai": 2.0, "anthropic": 2.0,
    },
    AICategory.CLI_TOOL: {
        "cli": 2.0, "command-line": 2.0, "terminal": 1.5, "shell": 1.0,
        "tui": 1.5, "console": 1.0,
    },
}

# Global AI-relevance keywords (any category)
_AI_SIGNALS: dict[str, float] = {
    "ai": 2.0, "artificial-intelligence": 3.0, "ml": 2.0, "machine-learning": 3.0,
    "deep-learning": 3.0, "neural": 2.0, "nlp": 2.5, "natural-language": 2.5,
    "computer-vision": 2.5, "generative": 2.5, "generative-ai": 3.0,
    "llm": 3.0, "gpt": 2.5, "transformer": 2.5, "diffusion": 2.0,
    "chatbot": 2.0, "conversational": 2.0, "openai": 2.0, "anthropic": 2.0,
    "huggingface": 2.5, "langchain": 3.0, "llamaindex": 3.0,
    "pytorch": 2.5, "tensorflow": 2.5, "onnx": 2.0, "cuda": 1.5,
}

# Non-AI penalty keywords — reduce score for repos that match AI keywords
# but are clearly not AI tools
_PENALTY_KEYWORDS: dict[str, float] = {
    "awesome-list": -2.0, "tutorial": -1.0, "course": -1.0, "learn": -0.5,
    "cheatsheet": -1.5, "interview": -1.5, "roadmap": -1.0,
}

_RELEVANCE_THRESHOLD = 3.0  # Minimum weighted score to be considered AI-related


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens, preserving hyphenated terms."""
    text = text.lower()
    # Split on non-alphanumeric except hyphens
    tokens = re.findall(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", text)
    return tokens


class KeywordClassifier(BaseClassifier):
    """Fast, zero-dependency classifier using curated keyword taxonomy."""

    async def classify(self, tool: AITool) -> ClassificationResult:
        """Classify a tool based on keyword matching."""
        text_sources = [
            tool.name,
            tool.description,
            " ".join(tool.topics),
            tool.author,
        ]
        all_text = " ".join(text_sources)
        tokens = _tokenize(all_text)
        token_set = set(tokens)

        # Score per category
        category_scores: dict[AICategory, float] = defaultdict(float)
        for category, keywords in _TAXONOMY.items():
            for keyword, weight in keywords.items():
                kw_tokens = set(_tokenize(keyword))
                if kw_tokens & token_set:
                    category_scores[category] += weight
                # Also check substring matching for compound terms
                elif keyword in all_text.lower():
                    category_scores[category] += weight * 0.8

        # Global AI relevance score
        ai_score = 0.0
        for keyword, weight in _AI_SIGNALS.items():
            kw_tokens = set(_tokenize(keyword))
            if kw_tokens & token_set:
                ai_score += weight
            elif keyword in all_text.lower():
                ai_score += weight * 0.8

        # Apply penalties
        for keyword, penalty in _PENALTY_KEYWORDS.items():
            if keyword in all_text.lower():
                ai_score += penalty  # penalty is negative

        # Also add top category score to AI score (being in a category = AI relevant)
        top_category_score = max(category_scores.values()) if category_scores else 0.0
        ai_score += top_category_score * 0.5

        # Normalize to 0-1
        max_possible = sum(max(v.values()) for v in _TAXONOMY.values()) * 0.5 + 15.0
        relevance = min(max(ai_score / max_possible, 0.0), 1.0)

        # Determine best category
        is_ai = ai_score >= _RELEVANCE_THRESHOLD
        if category_scores:
            best_category = max(category_scores, key=lambda k: category_scores[k])
            if category_scores[best_category] < 1.5:
                best_category = AICategory.OTHER
        else:
            best_category = AICategory.OTHER

        # Build summary
        summary = tool.description[:120] if tool.description else tool.name

        return ClassificationResult(
            is_ai_tool=is_ai,
            category=best_category,
            relevance_score=round(relevance, 3),
            summary=summary,
        )
