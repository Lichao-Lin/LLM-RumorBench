"""python -m rumorbench INPUT --labels LABEL ... --output DIRECTORY"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd

from . import __version__
from .evaluation import evaluate


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    raise ValueError("Input must be .csv or .xlsx")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Full label universe; ordered low to high only with --ordinal")
    parser.add_argument("--ordinal", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory; existing output will not be overwritten")
    args = parser.parse_args()
    try:
        report = evaluate(read_table(args.input), args.labels, ordinal=args.ordinal,
                          seed=args.seed, resamples=args.resamples)
        report["provenance"] = {
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "software": {"rumorbench": __version__, "python": platform.python_version(),
                         **{p: importlib.metadata.version(p)
                            for p in ("numpy", "pandas", "scipy", "scikit-learn", "openpyxl")}},
        }
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        rows = []
        for scope in ("available_case_metrics", "common_case_metrics"):
            for condition, metrics in report[scope].items():
                rows.append({"scope": scope, "condition": condition,
                             **{k: v for k, v in metrics.items() if k != "confusion_matrix"}})
        pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
        pd.DataFrame(report["baseline_comparisons"]).to_csv(
            args.output / "comparisons.csv", index=False)
        print(f"Saved {args.output}; common items: {report['coverage']['common_complete']}")
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
