"""
IBM Granite 4.0 1B Speech engine.

Uses the ``AutoModelForSpeechSeq2Seq`` model from HuggingFace transformers
for efficient on-device speech recognition (1B parameters, bfloat16).

Supported languages: en, fr, de, es, pt, ja.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .base import SpeechEngine

log = logging.getLogger(__name__)

GRANITE_REPO_ID = "ibm-granite/granite-4.0-1b-speech"


class GraniteSpeechEngine(SpeechEngine):
    """IBM Granite 4.0 1B Speech — compact 1B parameter ASR model."""

    def __init__(self) -> None:
        super().__init__()
        self._processor = None
        self._device: str = "cuda"

    # ── Abstract interface ───────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "granite"

    @property
    def vram_estimate_gb(self) -> float:
        return 3.0

    def load(self, model_path: str, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self._device = device
        granite_dir = os.path.join(model_path, "granite")

        # Download if not present locally
        if not os.path.isdir(granite_dir) or not os.path.isfile(
            os.path.join(granite_dir, "config.json")
        ):
            log.info("Granite model not found at %s — downloading…", granite_dir)
            from dictator.model_downloader import download_model
            download_model("granite", model_path)

        log.info("Loading Granite 4.0 1B Speech from %s", granite_dir)

        self._processor = AutoProcessor.from_pretrained(granite_dir)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            granite_dir,
            device_map=device if device == "cuda" else "cpu",
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        log.info("Granite 4.0 1B Speech loaded on %s", device)

    def _transcribe_impl(self, audio_16k: np.ndarray, language: str) -> str:
        import torch

        user_prompt = "<|audio|>can you transcribe the speech into a written format?"
        chat = [{"role": "user", "content": user_prompt}]
        prompt = self._processor.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
        )

        model_inputs = self._processor(
            prompt, audio_16k, device=self._device, return_tensors="pt",
        )
        # Move input tensors to model device
        model_inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v
                        for k, v in model_inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(**model_inputs, max_new_tokens=512)

        # Decode only the generated tokens (skip the input prompt tokens)
        input_len = model_inputs["input_ids"].shape[-1]
        text = self._processor.tokenizer.decode(
            output_ids[0, input_len:], skip_special_tokens=True,
        )
        return text.strip()

    def unload(self) -> None:
        self._processor = None
        self._release_model()
