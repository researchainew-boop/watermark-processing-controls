"""Cheap synthetic checks; no source data, models, or GPU are consumed."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from phase6_native_baseline import negative_cut, paired_metrics


class BaselineChecks(unittest.TestCase):
    def test_negative_cut_is_strict_and_uses_finite_sample_rank(self):
        cut, rank = negative_cut(np.arange(800))
        self.assertEqual(rank, 761)
        self.assertEqual(cut, 760)
        self.assertEqual(int((np.arange(800) > cut).sum()), 39)
        tied_cut, _ = negative_cut(np.zeros(800))
        self.assertEqual(int((np.zeros(800) > tied_cut).sum()), 0)
        with self.assertRaises(ValueError):
            negative_cut([0, np.nan])
        with self.assertRaises(ValueError):
            negative_cut(np.arange(5), .001)

    def test_perfect_scores_keep_paired_bootstrap_and_locked_decisions(self):
        frame = pd.DataFrame({"source_id": np.arange(40), "label": [0]*20+[1]*20,
            "collection": ["real"]*20+["AI"]*20, "native": [.1]*20+[.9]*20,
            "common": [.1]*20+[.9]*20})
        cuts = {k: {"threshold_logit": .5} for k in ["native", "common"]}
        report = paired_metrics(frame, cuts, replicates=20)
        self.assertEqual(report["paired_native_minus_common_AUC"], 0)
        self.assertEqual(report["paired_native_minus_common_AUC_ci95"], [0, 0])
        for mode in ["native", "common"]:
            self.assertEqual(report["metrics"][mode]["AUC"], 1)
            self.assertEqual(report["metrics"][mode]["TPR"], 1)
            self.assertEqual(report["metrics"][mode]["FPR"], 0)


if __name__ == "__main__":
    unittest.main()
