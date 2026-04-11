"""
Model downloader using huggingface_hub.

Downloads Cohere Transcribe and IBM Granite Speech models
from HuggingFace Hub to local storage.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

# ── Model constants (single source of truth) ─────────────────────────────────

COHERE_REPO_ID = "CohereLabs/cohere-transcribe-03-2026"
GRANITE_REPO_ID = "ibm-granite/granite-4.0-1b-speech"

_ENGINE_REPO_MAP = {
    "cohere": COHERE_REPO_ID,
    "granite": GRANITE_REPO_ID,
}


def download_model(engine_name: str, model_path: str) -> int:
    """Download model files for *engine_name* to *model_path*/<engine_name>.

    Returns 0 on success, 1 on failure.
    """
    repo_id = _ENGINE_REPO_MAP.get(engine_name)
    if repo_id is None:
        print(f"ERROR: Unknown engine '{engine_name}'. Choose from: {list(_ENGINE_REPO_MAP)}")
        return 1

    target_dir = os.path.join(model_path, engine_name)
    os.makedirs(target_dir, exist_ok=True)

    # Check if already downloaded
    if model_ready(engine_name, model_path):
        print(f"{engine_name.capitalize()} model already present in {target_dir} — skipping download.")
        return 0

    try:
        import huggingface_hub
    except ImportError:
        print("ERROR: huggingface-hub is required for model downloads.")
        print("Install it: pip install huggingface-hub")
        return 1

    print(f"Downloading {engine_name} model from {repo_id} to {target_dir}...")
    try:
        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_files_only=False,
        )
        print(f"{engine_name.capitalize()} model download complete.")
        return 0
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "Repository Not Found" in msg:
            print(f"ERROR: Repo not found or access denied: {exc}")
        else:
            print(f"ERROR: Download failed: {exc}")
        return 1


def model_ready(engine_name: str, model_path: str) -> bool:
    """Return True if the model files for *engine_name* exist."""
    engine_dir = os.path.join(model_path, engine_name)
    return os.path.isdir(engine_dir) and os.path.isfile(
        os.path.join(engine_dir, "config.json")
    )
