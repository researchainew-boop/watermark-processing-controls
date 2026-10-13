"""Synthetic verification only: no final or calibration research data are read."""
from __future__ import annotations

import copy
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

import phase6_fusion as fusion


QUERIES = ("0000000000000001", "0011001100110011", "1111111100000000")


@contextmanager
def scratch():
    # The managed Windows sandbox's redirected TEMP does not permit files
    # below a newly created subdirectory. Use the authorized workspace.
    parent = fusion.ROOT/"outputs"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent, prefix="phase6_unit_") as directory:
        path = Path(directory).resolve()
        if not path.is_relative_to(parent.resolve()):
            raise RuntimeError("Synthetic cleanup target escaped its workspace parent")
        yield path


def synthetic_frame(sources=24, conditions=("identity", "jpeg50")):
    rng = np.random.default_rng(1717)
    rows = []
    for label in (0, 1):
        for si in range(sources):
            sid = f"{label}-{si:04d}"
            collection = f"{'AI' if label else 'photo'}{si % 2}"
            for condition in conditions:
                covers = [("unmarked", "")] + ([(d, p) for d in fusion.METHODS for p in QUERIES] if label else [])
                for mark, payload in covers:
                    for detector in fusion.METHODS:
                        if mark not in ("unmarked", detector):
                            continue
                        image_id = f"{sid}-{condition}-{mark}-{payload}"
                        z = (-1.0 if label else -2.0) + .4*np.sin(si+len(condition))
                        for query in QUERIES:
                            agreement = .94 if payload == query else .5 + rng.uniform(-.2, .2)
                            margin = (agreement-.5)*3 + rng.uniform(-.04, .04)
                            features = dict(zip(fusion.FEATURES, [agreement, max(0, agreement-.05), margin,
                                                  margin-.1, abs(margin)+.2, .8+rng.uniform(-.15, .15),
                                                  .7+rng.uniform(-.1, .1), int(payload == query),
                                                  int(payload == query), int(payload == query), agreement]))
                            rows.append({"image_id": image_id, "source_id": sid, "role": "fit", "collection": collection,
                                         "label": label, "watermark_method": mark, "embedded_payload": payload,
                                         "condition": condition, "detector": detector, "query_payload": query,
                                         "resnet_logit": z, "clip_logit": z+.15, **features})
    return pd.DataFrame(rows)


def data_from_frame(frame):
    frame = frame.copy()
    frame["is_unmarked"] = frame.watermark_method.eq("unmarked")
    return fusion.view(frame, "trustmark", "resnet")


class FusionVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = synthetic_frame()
        cls.data = data_from_frame(cls.frame)
        cls.model = fusion.fit_correction(cls.data, 2., 1., "primary")

    def test_leading_zero_queries_and_exact_grouping(self):
        with scratch() as directory:
            path = Path(directory)/"features.csv"
            self.frame.sample(frac=1, random_state=1).to_csv(path, index=False)
            frame, queries = fusion.read_features(path, ("fit",))
            self.assertEqual(queries, sorted(QUERIES))
            self.assertEqual(frame.embedded_payload[frame.watermark_method.ne("unmarked")].str.len().min(), 16)
            self.assertEqual(len(fusion.view(frame, "trustmark", "resnet")["meta"])*3, len(frame[frame.detector.eq("trustmark")]))
            bad = self.frame.copy(); bad.loc[0, "role"] = "tune"; bad.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "multiple roles"):
                fusion.read_features(path)
            bad = pd.concat([self.frame, self.frame.iloc[:1]], ignore_index=True); bad.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "unique queries"):
                fusion.read_features(path)

    def test_no_metadata_prediction_and_bounded_correction(self):
        self.assertTrue(self.model["eligible"], self.model["optimizer"])
        s = fusion.score(self.data, self.model)
        self.assertTrue(np.all(s >= self.data["z"]))
        self.assertTrue(np.all(s <= self.data["z"]+2))
        altered = copy.deepcopy(self.data)
        for name in ("source_id", "role", "collection", "condition", "embedded_payload"):
            altered["meta"][name] = "irrelevant"
        altered["y"] = 1-altered["y"]
        np.testing.assert_array_equal(s, fusion.score(altered, self.model))
        np.testing.assert_array_equal(self.data["z"], fusion.score(self.data, {"kind": "probe"}))
        self.assertEqual(fusion.design(self.data, self.model).shape[-1], len(fusion.FEATURES)+2)

    def test_maximum_query_includes_unmarked_controls(self):
        data = copy.deepcopy(self.data)
        ai = fusion.FEATURES.index("agreement")
        data["features"][0, :, ai] = [.1, .2, .95]
        model = {"kind": "watermark_only", "feature": "agreement"}
        self.assertEqual(fusion.score(data, model)[0], .95)
        np.testing.assert_array_equal(fusion.query_max(np.array([[1., 8., 3.]]))[1], [1])

    def test_source_weights_and_pair_identity(self):
        w = fusion.weights(self.data)
        self.assertAlmostEqual(w[self.data["y"] == 0].sum(), .5)
        self.assertAlmostEqual(w[self.data["y"] == 1].sum(), .5)
        meta = self.data["meta"]
        by_source = pd.Series(w).groupby(meta.source_id).sum()
        self.assertAlmostEqual(by_source.max(), by_source.min())
        marked, unmarked, pw = fusion.pairs(self.data)
        self.assertAlmostEqual(pw.sum(), 1.)
        np.testing.assert_array_equal(meta.loc[marked, "source_id"], meta.loc[unmarked, "source_id"])
        np.testing.assert_array_equal(meta.loc[marked, "condition"], meta.loc[unmarked, "condition"])
        broken = copy.deepcopy(self.data)
        broken["meta"].loc[unmarked[0], "condition"] = "unpaired"
        with self.assertRaisesRegex(ValueError, "lacks an unmarked pair"):
            fusion.pairs(broken)

    def test_analytic_gradient_and_deterministic_fit(self):
        objective = fusion.correction_objective(self.data, self.model)
        theta = np.asarray(self.model["parameters"])+np.linspace(.001, .013, len(self.model["parameters"]))
        _, gradient = objective(theta)
        step = 1e-6
        numerical = []
        for i in range(len(theta)):
            delta = np.zeros_like(theta); delta[i] = step
            numerical.append((objective(theta+delta)[0]-objective(theta-delta)[0])/(2*step))
        np.testing.assert_allclose(gradient, numerical, rtol=2e-4, atol=2e-7)
        again = fusion.fit_correction(self.data, 2., 1., "primary")
        self.assertTrue(again["eligible"])
        np.testing.assert_array_equal(self.model["parameters"], again["parameters"])

    def test_baselines_are_finite_and_fit_only(self):
        static = fusion.fit_logistic(self.data)
        models = [static, fusion.fit_logistic(self.data, full=True), fusion.fit_reliability(self.data, static), fusion.fit_capped(self.data),
                  {"kind": "hard_gate", "static": static}, {"kind": "simple_confidence"}]
        for model in models:
            self.assertTrue(np.isfinite(fusion.score(self.data, model)).all(), model["kind"])

    def test_source_thresholds_and_ties(self):
        meta = pd.DataFrame({"source_id": np.repeat(np.arange(40).astype(str), 2), "label": 0, "collection": "photos"})
        scores = np.repeat(np.arange(40), 2).astype(float)
        empirical = fusion.empirical_threshold(meta, scores)
        self.assertEqual(sum(scores[::2] >= empirical), 2)
        conformal = fusion.conformal_cut(meta, scores)
        self.assertEqual(conformal["threshold"], 38.)
        self.assertEqual(sum(scores[::2] > conformal["threshold"]), 1)
        ties = fusion.conformal_cut(meta, np.zeros(len(meta)))
        self.assertEqual(sum(np.zeros(len(meta)) > ties["threshold"]), 0)
        small = fusion.conformal_cut(meta.iloc[:20], scores[:20])
        self.assertIsNone(small["threshold"])

    def test_zero_event_fixed_quota_harm_bound(self):
        def run(n):
            data = {"meta": pd.DataFrame({"source_id": np.arange(n).astype(str), "collection": np.asarray([f"family{i % 3}" for i in range(n)])}),
                    "y": np.ones(n), "unmarked": np.ones(n, bool)}
            return fusion.harmful_switch_assessment(data, np.ones(n, bool), np.ones(n, bool))
        large, small = run(600), run(200)
        self.assertAlmostEqual(large["pooled_zero_event_upper95"], 1-.05**(1/600))
        self.assertTrue(large["finite_sample_supported"])
        self.assertTrue(large["per_collection_noninferiority_unresolved"])
        self.assertFalse(small["finite_sample_supported"])

    def test_tune_unmarked_group_constraint_and_fallback(self):
        baseline = fusion.empirical_threshold(self.data["meta"], self.data["z"])
        cut, report = fusion.eligible_tune(self.data, {"kind": "probe"}, baseline)
        self.assertEqual(cut, baseline)
        self.assertTrue(report["empirically_feasible"])
        self.assertTrue(all(abs(x["tpr_minus_probe"]) == 0 for x in report["unmarked_groups"]))
        altered = copy.deepcopy(self.data)
        altered["features"][altered["unmarked"], :, fusion.FEATURES.index("agreement")] = 0.
        _, bad = fusion.eligible_tune(altered, {"kind": "watermark_only", "feature": "agreement"}, baseline)
        self.assertFalse(bad["empirically_feasible"])

    def test_bootstrap_is_paired_and_repeatable(self):
        baseline = self.data["z"] > -.8
        values = {"probe": baseline, "primary": baseline.copy()}
        first = fusion.source_bootstrap(self.data, values, baseline, 50, 7)
        second = fusion.source_bootstrap(self.data, values, baseline, 50, 7)
        self.assertEqual(first, second)
        self.assertEqual(first["primary"]["marked_tpr"]["minus_probe"]["ci95"], [0., 0.])
        self.assertEqual(len(first["primary"]["marked_shares_secondary"]), 5)

    def test_receipt_resume_checks_inputs_and_outputs(self):
        with scratch() as directory:
            out = Path(directory)
            fusion.write_json(out/"models.json", {"example": 1})
            receipt = {"schema": fusion.SCHEMA, "script_sha256": fusion.digest(fusion.__file__), "development_features_sha256": "featurehash",
                       "outputs_sha256": {"models.json": fusion.digest(out/"models.json")}}
            fusion.write_json(out/"fit_complete.json", receipt)
            self.assertEqual(fusion.verify_complete(out, "fit_complete.json", {"development_features_sha256": "featurehash"}), receipt)
            with self.assertRaisesRegex(RuntimeError, "input changed"):
                fusion.verify_complete(out, "fit_complete.json", {"development_features_sha256": "different"})
            fusion.write_json(out/"models.json", {"example": 2})
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                fusion.verify_complete(out, "fit_complete.json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
