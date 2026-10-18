import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from kappa import simple_kappa_test
from rumorbench.__main__ import read_table
from rumorbench.evaluation import CONDITIONS, TRUTH, evaluate, holm, score, validate

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["0", "1", "2"]


def fixture():
    return read_table(ROOT / "examples/synthetic.csv")


class EvaluationTests(unittest.TestCase):
    def test_known_metrics(self):
        result = score(["0", "0", "1", "1"], ["0", "1", "1", "1"], ["0", "1"])
        self.assertEqual(result["accuracy"], .75)
        self.assertEqual(result["kappa"], .5)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + .8) / 2)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])

    def test_ordinal_uses_declared_universe(self):
        result = score(["low", "high"], ["low", "middle"],
                       ["low", "middle", "high"], ordinal=True)
        self.assertAlmostEqual(result["linear_kappa"], .5)
        self.assertAlmostEqual(result["spearman"], 1)
        self.assertIsNone(score(["0", "1"], ["0", "1"], LABELS)["spearman"])

    def test_degenerate_is_null_not_zero(self):
        result = score(["0", "0"], ["0", "0"], LABELS, ordinal=True)
        self.assertEqual(result["accuracy"], 1)
        self.assertIsNone(result["kappa"])
        self.assertIsNone(result["spearman"])
        self.assertIsNone(score([], [], LABELS)["accuracy"])
        json.dumps(result, allow_nan=False)

    def test_preserves_input_and_common_cohort(self):
        frame = fixture()
        original = frame.copy()
        result = evaluate(frame, LABELS, resamples=100)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(result["coverage"]["common_complete"], 11)
        self.assertEqual(result["available_case_metrics"]["Neither"]["n"], 12)
        self.assertTrue(all(v["n"] == 11 for v in result["common_case_metrics"].values()))
        self.assertTrue(all(v["n"] == 11 for v in result["baseline_comparisons"]))

    def test_order_and_seed_reproducibility(self):
        first = evaluate(fixture(), LABELS, seed=42, resamples=100)
        second = evaluate(fixture().iloc[::-1], LABELS, seed=42, resamples=100)
        self.assertEqual(first, second)

    def test_identity_has_zero_deltas_and_unit_p(self):
        frame = fixture()
        for condition in CONDITIONS:
            frame[condition] = frame[TRUTH]
        result = evaluate(frame, LABELS, resamples=100)
        for row in result["baseline_comparisons"]:
            self.assertEqual(row["accuracy_delta"], 0)
            self.assertEqual(row["ci95"], [0, 0])
            self.assertEqual(row["mcnemar_exact_p"], 1)
        self.assertEqual(result["factorial_accuracy_contrasts"]["interaction"]["estimate"], 0)

    def test_mcnemar_and_factorial_known_contrast(self):
        frame = fixture().iloc[:6].copy()
        frame[TRUTH] = "0"
        frame["Neither"] = "1"
        frame["web search"] = "0"
        frame["deep think"] = "1"
        frame["Both"] = "0"
        result = evaluate(frame, LABELS, resamples=100)
        row = result["baseline_comparisons"][0]
        self.assertEqual((row["wins"], row["losses"]), (6, 0))
        self.assertEqual(row["mcnemar_exact_p"], 2 / 64)
        self.assertEqual(row["holm_p"], 3 * 2 / 64)
        effects = result["factorial_accuracy_contrasts"]
        self.assertEqual(effects["web_main"]["estimate"], 1)
        self.assertEqual(effects["think_main"]["estimate"], 0)
        self.assertEqual(effects["interaction"]["estimate"], 0)

    def test_holm_unsorted_and_monotonic(self):
        np.testing.assert_allclose(holm([.04, .001, .03]), [.06, .003, .06])

    def test_missing_and_invalid_inputs(self):
        frame = fixture()
        for bad in (frame.drop(columns="sample_id"), frame.iloc[:0],
                    pd.concat([frame, frame.iloc[:1]])):
            with self.assertRaises(ValueError):
                evaluate(bad, LABELS, resamples=100)
        for column, value in (("sample_id", ""), (TRUTH, ""), ("Both", "invalid")):
            bad = frame.copy()
            bad.loc[0, column] = value
            with self.assertRaises(ValueError):
                evaluate(bad, LABELS, resamples=100)
        for labels in (["0"], ["0", "0"], ["", "1"]):
            with self.assertRaises(ValueError):
                validate(frame, labels)

    def test_no_common_items_skips_inference(self):
        frame = fixture()
        frame["Both"] = ""
        result = evaluate(frame, LABELS, resamples=100)
        self.assertEqual(result["coverage"]["common_complete"], 0)
        self.assertEqual(result["baseline_comparisons"], [])
        self.assertIsNone(result["common_case_metrics"]["Both"]["accuracy"])

    def test_invalid_bootstrap_settings(self):
        for kwargs in ({"resamples": 0}, {"resamples": True}, {"seed": -1}):
            with self.assertRaises(ValueError):
                evaluate(fixture(), LABELS, **kwargs)

    def test_legacy_row_alignment_and_three_outputs(self):
        # Separately dropping each column would shift the last pair.
        frame = pd.DataFrame({"a": [0, None, 1, 1], "b": [0, 0, None, 1],
                              "c": [0, 1, 2, 2], "d": [0, 1, 1, 2]})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "legacy.xlsx"
            frame.to_excel(path, index=False)
            result = simple_kappa_test(path, LABELS)
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0], 1)
            process = subprocess.run([sys.executable, "kappa.py", str(path),
                                      "--ordinal-labels", *LABELS],
                                     cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)

    def test_cli_artifacts_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "result"
            command = [sys.executable, "-m", "rumorbench", "examples/synthetic.csv",
                       "--labels", *LABELS, "--ordinal", "--resamples", "100",
                       "--output", str(out)]
            process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads((out / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(len(result["provenance"]["input_sha256"]), 64)
            self.assertEqual(len(pd.read_csv(out / "metrics.csv")), 8)
            self.assertEqual(len(pd.read_csv(out / "comparisons.csv")), 3)
            again = subprocess.run(command, cwd=ROOT, capture_output=True)
            self.assertNotEqual(again.returncode, 0)


if __name__ == "__main__":
    unittest.main()
