from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np


class Reranker(Protocol):
    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray: ...

    def metadata(self) -> dict[str, Any]: ...

    def release(self) -> None: ...


class TransformerCrossEncoderReranker:
    def __init__(
        self,
        *,
        model_name: str,
        revision: str,
        cache_dir: str | Path,
        device: str,
        batch_size: int,
        max_length: int,
        use_half_on_cuda: bool,
        seed: int,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.model_name = model_name
        self.revision = revision
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_half_on_cuda = use_half_on_cuda and device == "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=str(self.cache_dir),
            local_files_only=True,
        )
        dtype = torch.float16 if self.use_half_on_cuda else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=str(self.cache_dir),
            local_files_only=True,
            torch_dtype=dtype,
        )
        self.model.to(device)
        self.model.eval()
        self._torch = torch

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.empty((0,), dtype=np.float32)
        scores: list[np.ndarray] = []
        for offset in range(0, len(pairs), self.batch_size):
            batch = pairs[offset : offset + self.batch_size]
            inputs = self.tokenizer(
                list(batch),
                padding=True,
                truncation="longest_first",
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                logits = self.model(**inputs, return_dict=True).logits.reshape(-1)
            scores.append(logits.float().cpu().numpy())
        return np.concatenate(scores).astype(np.float32, copy=False)

    def token_lengths(
        self, pairs: Sequence[tuple[str, str]], *, truncation: bool
    ) -> list[int]:
        return [
            len(
                self.tokenizer(
                    query,
                    passage,
                    truncation="longest_first" if truncation else False,
                    max_length=self.max_length,
                    add_special_tokens=True,
                )["input_ids"]
            )
            for query, passage in pairs
        ]

    def metadata(self) -> dict[str, Any]:
        config_path = Path(self.model.config._name_or_path) / "config.json"
        architecture = getattr(self.model.config, "architectures", None)
        return {
            "model_name": self.model_name,
            "resolved_revision": self.revision,
            "architecture": architecture,
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "tokenizer_name": self.tokenizer.name_or_path,
            "native_max_sequence_length": int(self.tokenizer.model_max_length),
            "max_sequence_length": self.max_length,
            "device": self.device,
            "inference_precision": "float16" if self.use_half_on_cuda else "float32",
            "output_dtype": "float32",
            "batch_size": self.batch_size,
            "config_path": str(config_path) if config_path.exists() else None,
        }

    def release(self) -> None:
        del self.model
        del self.tokenizer
        if self.device == "cuda":
            self._torch.cuda.empty_cache()


def load_reranking_config(path: str | Path) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["model"]["name"] != "BAAI/bge-reranker-v2-m3":
        raise ValueError("Phase 7 baseline requires BAAI/bge-reranker-v2-m3")
    if config["model"]["precision"] not in {"float16", "float32"}:
        raise ValueError("unsupported reranker precision")
    depths = [int(value) for value in config["candidate_pool"]["rerank_depths"]]
    if depths != sorted(set(depths)) or not depths:
        raise ValueError("rerank_depths must be non-empty, unique, and sorted")
    if int(config["candidate_pool"]["default_depth"]) not in depths:
        raise ValueError("default_depth must be included in rerank_depths")
    return config
