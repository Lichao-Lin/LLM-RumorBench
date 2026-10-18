"""python -m rumorbench INPUT --labels LABEL ... --output DIRECTORY"""

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import platform
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from . import __version__
from .evaluation import evaluate


COMPARISON_COLUMNS = ["condition", "baseline", "n", "accuracy_delta", "ci95",
                      "wins", "losses", "mcnemar_exact_p", "holm_p"]


def read_table(path, payload=None):
    path = Path(path)
    payload = path.read_bytes() if payload is None else payload
    if path.suffix.lower() == ".csv":
        header = next(csv.reader(io.StringIO(payload.decode("utf-8-sig"))), [])
    elif path.suffix.lower() == ".xlsx":
        workbook = load_workbook(io.BytesIO(payload), read_only=True, data_only=True)
        try:
            header = list(next(workbook.worksheets[0].values, ()))
        finally:
            workbook.close()
    else:
        raise ValueError("Input must be .csv or .xlsx")
    if not header or len(set(header)) != len(header):
        raise ValueError("Input headers must be nonempty and unique")
    reader = pd.read_csv if path.suffix.lower() == ".csv" else pd.read_excel
    return reader(io.BytesIO(payload), dtype=str, keep_default_na=False)


def source_fingerprints():
    package = Path(__file__).resolve().parent
    return {path.name: hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for path in sorted(package.glob("*.py"))}


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
        payload = args.input.read_bytes()
        report = evaluate(read_table(args.input, payload), args.labels, ordinal=args.ordinal,
                          seed=args.seed, resamples=args.resamples)
        report["provenance"] = {
            "input_sha256": hashlib.sha256(payload).hexdigest(),
            "source_sha256_lf": source_fingerprints(),
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
        pd.DataFrame(report["baseline_comparisons"], columns=COMPARISON_COLUMNS).to_csv(
            args.output / "comparisons.csv", index=False)
        print(f"Saved {args.output}; common items: {report['coverage']['common_complete']}")
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
