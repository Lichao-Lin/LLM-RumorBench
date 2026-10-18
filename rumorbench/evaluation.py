"""Validated, paired analysis. No model API calls or invented paper results."""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

CONDITIONS = ("Neither", "web search", "deep think", "Both")
TRUTH = "correct answer"


def validate(data, labels):
    """One row per independent item, one response per condition."""
    required = ["sample_id", TRUTH, *CONDITIONS]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Provide at least two distinct labels in the intended order")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("Labels must be nonempty strings")
    frame = data.copy()
    if frame.empty:
        raise ValueError("Input contains no samples")
    ids = frame["sample_id"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("sample_id must be nonempty and unique; do not pool repeated runs")
    frame["sample_id"] = ids
    for column in [TRUTH, *CONDITIONS]:
        values = frame[column].astype("string").str.strip().replace("", pd.NA)
        unknown = set(values.dropna()) - set(labels)
        if unknown:
            raise ValueError(f"Unknown labels in {column}: {sorted(unknown)}")
        frame[column] = values
    if frame[TRUTH].isna().any():
        raise ValueError("Ground truth cannot be missing")
    return frame.sort_values("sample_id").reset_index(drop=True)


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def score(truth, prediction, labels, ordinal=False):
    """Null denotes undefined statistics, never zero agreement."""
    truth, prediction = list(truth), list(prediction)
    if not truth:
        return {"n": 0, "accuracy": None, "macro_f1": None, "kappa": None,
                "linear_kappa": None, "quadratic_kappa": None, "spearman": None,
                "confusion_matrix": [[0] * len(labels) for _ in labels]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = {
            "n": len(truth),
            "accuracy": float(np.mean(np.array(truth) == np.array(prediction))),
            "macro_f1": float(f1_score(truth, prediction, labels=labels,
                                       average="macro", zero_division=0)),
            "kappa": finite_or_none(cohen_kappa_score(truth, prediction, labels=labels)),
            "linear_kappa": None, "quadratic_kappa": None, "spearman": None,
            "confusion_matrix": confusion_matrix(truth, prediction, labels=labels).tolist(),
        }
        if ordinal:
            for weights in ("linear", "quadratic"):
                result[f"{weights}_kappa"] = finite_or_none(
                    cohen_kappa_score(truth, prediction, labels=labels, weights=weights))
            ranks = {label: index for index, label in enumerate(labels)}
            a, b = [ranks[x] for x in truth], [ranks[x] for x in prediction]
            if len(set(a)) > 1 and len(set(b)) > 1:
                result["spearman"] = finite_or_none(spearmanr(a, b).statistic)
    return result


def mean_interval(values, rng, resamples):
    """Percentile bootstrap of item-level contrasts; pairing is retained."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    # Chunk draws to avoid allocating resamples * number_of_items at once.
    means = []
    for start in range(0, resamples, 100):
        count = min(100, resamples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means.extend(values[indices].mean(axis=1))
    return np.quantile(means, [0.025, 0.975]).tolist()


def holm(pvalues):
    """Holm step-down adjustment for the complete planned test family."""
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def evaluate(data, labels, *, ordinal=False, seed=2026, resamples=5000):
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
        raise ValueError("resamples must be an integer >= 100")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    frame = validate(data, labels)
    common = frame.dropna(subset=list(CONDITIONS))
    result = {
        "schema_version": 1,
        "settings": {"labels": labels, "ordinal": ordinal, "seed": seed,
                     "resamples": resamples, "ci": "95% percentile bootstrap"},
        "coverage": {"total": len(frame), "common_complete": len(common),
                     "missing_predictions": {c: int(frame[c].isna().sum()) for c in CONDITIONS}},
        "available_case_metrics": {}, "common_case_metrics": {},
        "baseline_comparisons": [], "factorial_accuracy_contrasts": {},
        "warnings": ["Exploratory extension; not a reproduction of published numeric results.",
                     "Rows must be independent items. Repeated runs/events require clustered analysis.",
                     "Complete-case estimates may be biased by selective missing responses.",
                     "Bootstrap intervals are pointwise, not multiplicity-adjusted; tiny samples are unstable."],
    }
    for condition in CONDITIONS:
        available = frame.dropna(subset=[condition])
        result["available_case_metrics"][condition] = score(
            available[TRUTH], available[condition], labels, ordinal)
        result["common_case_metrics"][condition] = score(
            common[TRUTH], common[condition], labels, ordinal)
    if len(common) < 2:
        result["warnings"].append("Fewer than two common items; paired inference is unavailable.")
        return result
    rng = np.random.default_rng(seed)
    correct = {c: (common[c] == common[TRUTH]).to_numpy(dtype=int) for c in CONDITIONS}
    base = correct["Neither"]
    for condition in CONDITIONS[1:]:
        treatment = correct[condition]
        wins = int(((treatment == 1) & (base == 0)).sum())
        losses = int(((treatment == 0) & (base == 1)).sum())
        delta = treatment - base
        result["baseline_comparisons"].append({
            "condition": condition, "baseline": "Neither", "n": len(common),
            "accuracy_delta": float(delta.mean()),
            "ci95": mean_interval(delta, rng, resamples),
            "wins": wins, "losses": losses,
            "mcnemar_exact_p": float(binomtest(wins, wins + losses).pvalue)
            if wins + losses else 1.0,
        })
    adjusted = holm([row["mcnemar_exact_p"] for row in result["baseline_comparisons"]])
    for row, pvalue in zip(result["baseline_comparisons"], adjusted):
        row["holm_p"] = pvalue
    neither, web, think, both = (correct[c] for c in CONDITIONS)
    contrasts = {
        "web_main": ((web - neither) + (both - think)) / 2,
        "think_main": ((think - neither) + (both - web)) / 2,
        "interaction": both - web - think + neither,
    }
    for name, values in contrasts.items():
        result["factorial_accuracy_contrasts"][name] = {
            "estimate": float(values.mean()), "ci95": mean_interval(values, rng, resamples),
            "n": len(common),
        }
    return result
