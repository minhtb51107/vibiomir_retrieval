from __future__ import annotations

import math
from typing import Any, Sequence


CONTRACT_FIELDS = (
    "model", "revision", "tokenizer_revision", "batch_size", "max_length",
    "precision", "eval_mode", "padding", "truncation", "score_extraction",
    "input_order",
)


def validate_score_equivalence(
    reference: Sequence[float], observed: Sequence[float], keys: Sequence[str], *,
    measured_absolute_tolerance: float,
    expected_contract: dict[str, Any], actual_contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate an FP16 replay without allowing config or rank-order drift."""
    contract_mismatches = {
        field: {"expected": expected_contract.get(field), "actual": actual_contract.get(field)}
        for field in CONTRACT_FIELDS if expected_contract.get(field) != actual_contract.get(field)
    }
    if len(reference) != len(observed) or len(reference) != len(keys):
        return {"passed": False, "failure": "length_mismatch", "contract_mismatches": contract_mismatches}
    differences = [abs(float(left)-float(right)) for left,right in zip(reference,observed,strict=True)]
    finite = all(math.isfinite(float(value)) for value in [*reference,*observed])
    reference_order = sorted(range(len(keys)),key=lambda i:(-float(reference[i]),str(keys[i])))
    observed_order = sorted(range(len(keys)),key=lambda i:(-float(observed[i]),str(keys[i])))
    result={
        "checked":len(reference),"max_absolute_difference":max(differences,default=0.0),
        "measured_absolute_tolerance":float(measured_absolute_tolerance),
        "contract_mismatches":contract_mismatches,"finite":finite,
        "reference_order":reference_order,"observed_order":observed_order,
        "ordering_unchanged":reference_order==observed_order,
    }
    result["passed"]=not contract_mismatches and finite and result["max_absolute_difference"]<=measured_absolute_tolerance and result["ordering_unchanged"]
    return result
