from __future__ import annotations

import json
import random
import unicodedata
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


class Embedder(Protocol):
    dimension: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...

    def metadata(self) -> dict[str, Any]: ...


def _resolved_revision(model_name: str) -> str | None:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        reference = (
            Path(HF_HUB_CACHE)
            / f"models--{model_name.replace('/', '--')}"
            / "refs"
            / "main"
        )
        if reference.exists():
            return reference.read_text(encoding="utf-8").strip()

        from huggingface_hub import scan_cache_dir

        repos = [
            repo
            for repo in scan_cache_dir().repos
            if repo.repo_type == "model" and repo.repo_id == model_name
        ]
        revisions = [revision for repo in repos for revision in repo.revisions]
        if revisions:
            return max(revisions, key=lambda item: item.last_modified).commit_hash
    except Exception:
        return None
    return None


class SentenceTransformerEmbedder:
    def __init__(
        self,
        *,
        model_name: str,
        revision: str | None,
        device: str,
        batch_size: int,
        max_length: int,
        normalize_embeddings: bool,
        use_half_on_cuda: bool,
        seed: int,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        if model_name != "BAAI/bge-m3":
            raise ValueError("this baseline must use BAAI/bge-m3")
        actual_device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        if actual_device == "auto":
            actual_device = "cpu"
        if actual_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable in the installed PyTorch")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.model_name = model_name
        self.requested_revision = revision
        self.device = actual_device
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize_embeddings
        self.inference_precision = (
            "float16" if actual_device == "cuda" and use_half_on_cuda else "float32"
        )
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=actual_device,
        )
        self.native_max_length = int(self.model.max_seq_length)
        self.model.max_seq_length = max_length
        if actual_device == "cuda" and use_half_on_cuda:
            self.model.half()
        self.tokenizer = self.model.tokenizer
        dimension_getter = getattr(
            self.model,
            "get_embedding_dimension",
            self.model.get_sentence_embedding_dimension,
        )
        self.dimension = int(dimension_getter())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        normalized = [normalize_text(text) for text in texts]
        vectors = self.model.encode(
            normalized,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        values = np.asarray(vectors, dtype=np.float32)
        return l2_normalize(values) if self.normalize else values

    def metadata(self) -> dict[str, Any]:
        pooling = []
        for module in self.model.modules():
            if module.__class__.__name__ == "Pooling":
                for name in (
                    "cls_token",
                    "mean_tokens",
                    "max_tokens",
                    "mean_sqrt_len_tokens",
                    "weightedmean_tokens",
                    "lasttoken",
                ):
                    if getattr(module, f"pooling_mode_{name}", False):
                        pooling.append(name)
        resolved_revision = _resolved_revision(self.model_name)
        if not pooling and resolved_revision:
            try:
                from huggingface_hub.constants import HF_HUB_CACHE

                pooling_config = (
                    Path(HF_HUB_CACHE)
                    / f"models--{self.model_name.replace('/', '--')}"
                    / "snapshots"
                    / resolved_revision
                    / "1_Pooling"
                    / "config.json"
                )
                values = json.loads(pooling_config.read_text(encoding="utf-8"))
                pooling = [
                    name.removeprefix("pooling_mode_")
                    for name, enabled in values.items()
                    if name.startswith("pooling_mode_") and enabled is True
                ]
            except (OSError, json.JSONDecodeError):
                pass
        return {
            "model_name": self.model_name,
            "requested_revision": self.requested_revision,
            "resolved_revision": resolved_revision,
            "embedding_dimension": self.dimension,
            "tokenizer_name": getattr(self.tokenizer, "name_or_path", self.model_name),
            "native_max_sequence_length": self.native_max_length,
            "max_sequence_length": self.max_length,
            "pooling": pooling or ["model_defined"],
            "normalization": "L2" if self.normalize else "none",
            "inference_precision": self.inference_precision,
            "output_dtype": "float32",
            "device": self.device,
        }


def file_sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
