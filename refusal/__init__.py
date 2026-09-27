"""Refusal localization, layer freezing and weight-space repair under few-sample fine-tuning.

    import refusal

    stats = refusal.update_stats(A, B)          # participation ratio, top-2 energy, norms
    A2, B2 = refusal.remove_top_k(A, B, k=2)    # energy-ranked repair of one LoRA module
    refusal.is_refusal("I cannot help with that.")

The spectral core, the refusal scorers and the data selection need neither a GPU nor torch.
Torch, Transformers and PEFT are imported only when a model-facing name is first used.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .data import fingerprint, load_data, prepare
from .metrics import distinct_n, is_refusal, judge_llamaguard, summarize, transition_depth
from .spectrum import (aggregate_stats, calibrate_detector, concentration, factor_svd, remove_top_k,
                       threshold_at_fpr, update_stats)

# Model-facing names, imported on first attribute access (PEP 562).
_LAZY = {
    "load_model": "common",
    "generate": "common",
    "add_adapter": "train",
    "fit": "train",
    "lockstep_tokens": "patching",
    "lockstep_generate": "patching",
    "remove_top": "spectral",
    "remove_random": "spectral",
    "concentration_penalty": "spectral",
    "spectrum_stats": "spectral",
    "adapter_stats": "spectral",
    "compare_features": "probe",
}

__all__ = [
    "__version__",
    # weight-space repair and the spectral detector, numpy only
    "update_stats", "aggregate_stats", "factor_svd", "remove_top_k", "concentration",
    "threshold_at_fpr", "calibrate_detector",
    # scoring
    "is_refusal", "distinct_n", "summarize", "transition_depth", "judge_llamaguard",
    # data selection
    "prepare", "load_data", "fingerprint",
    # models, needs the [models] extra
    "load_model", "generate", "add_adapter", "fit", "lockstep_tokens", "lockstep_generate",
    "remove_top", "remove_random", "concentration_penalty", "spectrum_stats", "adapter_stats",
    "compare_features",
]


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
