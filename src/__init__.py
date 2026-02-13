"""
narremgen
=========

Framework for narrative and emotional text generation based on structured
narrative schemes (SN) and emotional dynamics (DE).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.9.5"
__author__ = "Rodolphe Priam"
__license__ = "MIT"
__email__ = "rpriam@gmail.com"

__all__ = [
    "LLMConnect",
    "generate_advice",
    "generate_mapping",
    "generate_context",
    "generate_narratives",
    "generate_narratives_batch",
    "build_corpus_for_variants",
    "run_llm_theme_pipeline",
    "run_one_variant_pipeline",
    "run_pipeline",
    "analyze_sn_de_distribution",
    "save_output",
    "safe_generate",
    "merge_and_filter",
    "renumerote_filtered",
    "audit_filtered",
    "validate_mapping",
    "quick_check_filtered",
]

_LAZY: dict[str, tuple[str, str]] = {
    "LLMConnect": ("llmcore", "LLMConnect"),
    "safe_chat_completion": ("llmcore", "safe_chat_completion"),
    "estimate_tokens": ("llmcore", "estimate_tokens"),
    "generate_advice": ("data", "generate_advice"),
    "generate_mapping": ("data", "generate_mapping"),
    "generate_context": ("data", "generate_context"),
    "generate_narratives": ("narratives", "generate_narratives"),
    "generate_narratives_batch": ("narratives", "generate_narratives_batch"),
    "run_pipeline": ("pipeline", "run_pipeline"),
    "build_corpus_for_variants": ("chapters", "build_corpus_for_variants"),
    "run_llm_theme_pipeline": ("themes", "run_llm_theme_pipeline"),
    "run_one_variant_pipeline": ("variants", "run_one_variant_pipeline"),
    "analyze_sn_de_distribution": ("analyzestats", "analyze_sn_de_distribution"),
    "save_output": ("utils", "save_output"),
    "safe_generate": ("utils", "safe_generate"),
    "merge_and_filter": ("utils", "merge_and_filter"),
    "renumerote_filtered": ("utils", "renumerote_filtered"),
    "audit_filtered": ("utils", "audit_filtered"),
    "validate_mapping": ("utils", "validate_mapping"),
    "quick_check_filtered": ("utils", "quick_check_filtered"),
}

_HINTS: dict[str, str] = {
    "openai": 'pip install ".[llm]"',
    "transformers": 'pip install ".[emotions]"',
    "torch": 'pip install ".[emotions]"',
    "vaderSentiment": 'pip install ".[emotions]"',
    "lexicalrichness": 'pip install ".[textstats]"',
    "matplotlib": 'pip install ".[plots]"',
    "reportlab": 'pip install ".[all]"',
    "docx": 'pip install ".[all]"',
}

def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module has no attribute {name!r}")

    module_name, attr = _LAZY[name]
    try:
        mod = import_module(f".{module_name}", __name__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None)
        if missing in _HINTS:
            raise ModuleNotFoundError(
                f"Optional dependency missing: {missing!r}. To enable this feature: {_HINTS[missing]}"
            ) from e
        raise
    except AttributeError as e:
        raise AttributeError(
            f"Public symbol {name!r} is mapped to {module_name}.{attr} but it does not exist."
        ) from e


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY.keys()))
