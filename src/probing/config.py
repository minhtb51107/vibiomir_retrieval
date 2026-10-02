from __future__ import annotations

from pathlib import Path

import yaml

from .models import ProbeConfig, SamplingRule


def load_probe_config(path: str | Path) -> ProbeConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    http = raw["http"]
    rules = tuple(
        SamplingRule(
            name=str(item["name"]),
            kind=str(item["kind"]),
            quota=int(item["quota"]),
            value=str(item["value"]) if item.get("value") is not None else None,
            inferred_language=(
                str(item["inferred_language"])
                if item.get("inferred_language") is not None
                else None
            ),
        )
        for item in raw["sampling_rules"]
    )
    maximum = int(raw["maximum_live_urls"])
    absolute = int(raw["absolute_safety_ceiling"])
    if maximum <= 0 or maximum > absolute or absolute > 200:
        raise ValueError("probe limits must satisfy 0 < maximum <= absolute <= 200")
    if sum(rule.quota for rule in rules) > maximum:
        raise ValueError("sampling rule quotas exceed maximum_live_urls")
    if any(rule.quota <= 0 for rule in rules):
        raise ValueError("sampling quotas must be positive")
    return ProbeConfig(
        input_path=str(raw["input_path"]),
        output_dir=str(raw["output_dir"]),
        seed=str(raw["seed"]),
        maximum_live_urls=maximum,
        absolute_safety_ceiling=absolute,
        batch_size=int(raw["batch_size"]),
        global_concurrency=int(http["global_concurrency"]),
        per_domain_concurrency=int(http["per_domain_concurrency"]),
        domain_delay_seconds=float(http["domain_delay_seconds"]),
        connect_timeout_seconds=float(http["connect_timeout_seconds"]),
        read_timeout_seconds=float(http["read_timeout_seconds"]),
        max_response_bytes=int(http["max_response_bytes"]),
        user_agent=str(http["user_agent"]),
        sampling_rules=rules,
    )
