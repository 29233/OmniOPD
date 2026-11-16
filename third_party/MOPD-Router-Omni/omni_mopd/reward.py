"""Diagnostic category metric only; upstream pure OPD skips training rewards."""
import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    match = re.search(r"Decision:\s*The target is\s+([^.\n]+)", solution_str, re.IGNORECASE)
    predicted = match.group(1).strip().lower() if match else ""
    return float(predicted == str(ground_truth).strip().lower())
