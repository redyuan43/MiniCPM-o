"""Helpers for importing the vendored MiniCPM-o duplex core."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = PROJECT_ROOT / "third_party" / "MiniCPM-o-Demo"


def ensure_vendor_path() -> None:
    if not VENDOR_ROOT.exists():
        raise FileNotFoundError(
            f"Vendored MiniCPM-o-Demo not found at {VENDOR_ROOT}. "
            "Clone it with scripts/setup_duplex_env.sh first."
        )
    vendor_str = str(VENDOR_ROOT)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)


def _patch_token2wav_float16_cache() -> None:
    import torch
    import stepaudio2.token2wav as token2wav_module

    token2wav_cls = token2wav_module.Token2wav
    if getattr(token2wav_cls, "_local_duplex_float16_patch", False):
        return

    original_prepare_prompt = token2wav_cls._prepare_prompt

    def _prepare_prompt_with_dtype_fix(self: Any, prompt_wav: str):
        cache = original_prepare_prompt(self, prompt_wav)
        if not getattr(self, "float16", False):
            return cache

        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = cache
        target_dtype = self.flow.spk_embed_affine_layer.weight.dtype
        spk_emb = spk_emb.to(dtype=target_dtype)
        prompt_mels = prompt_mels.to(dtype=target_dtype)
        return prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens

    token2wav_cls._prepare_prompt = _prepare_prompt_with_dtype_fix
    token2wav_cls._local_duplex_float16_patch = True


def import_duplex_core():
    ensure_vendor_path()
    _patch_token2wav_float16_cache()
    from core.processors import UnifiedProcessor  # type: ignore
    from core.schemas import DuplexConfig  # type: ignore

    return UnifiedProcessor, DuplexConfig
