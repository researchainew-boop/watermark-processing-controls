"""Guard-only checks on synthetic temporary files; no data, models, or GPU."""
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import phase6_native_bound as guard
import phase6_native_baseline as native


class GuardChecks(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1]/".tools/native-bound-tests"
        test_root.mkdir(parents=True,exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=test_root)
        self.root = Path(self.tmp.name)
        assert self.root.resolve().is_relative_to(test_root.resolve())
        self.addCleanup(self.tmp.cleanup)
        self.context = ExitStack(); self.addCleanup(self.context.close)
        self.out = self.root/"outputs/phase6/native_baseline"
        self.out.mkdir(parents=True)
        self.sources = self.root/"outputs/phase6/data_inventory/primary_source_manifest.csv"
        self.sources.parent.mkdir(parents=True)
        self.lockpath = self.root/"outputs/phase6/frozen_protocol.json"
        self.binding = self.root/"outputs/phase6/test_sources_locked.json"
        self.wrapper = self.root/"scripts/phase6_native_bound.py"
        self.wrapper.parent.mkdir(parents=True)
        self.wrapper.write_text("unchanged supplemental wrapper")
        previous = self.root/"scripts/native_baseline.py"; previous.write_text("unchanged prior driver")
        required = {"calibration":{"coco":400,"openimages":400},
                    "test":{"coco":200,"openimages":200,"BigGAN":200,"ADM":200,"VQDM":200}}
        rows = []
        for role,groups in required.items():
            for collection,n in groups.items():
                rows += [{"source_id":f"{role}-{collection}-{i:04d}","role":role,"split":role,
                          "collection":collection,"label":int(collection not in ["coco","openimages"])} for i in range(n)]
        self.write_csv(self.sources,rows)
        self.calpath = self.sources.with_name("calibration_source_manifest.csv")
        self.write_csv(self.calpath,[r for r in rows if r["role"]=="calibration"])
        assets = {".tools/extension-assets/test-model.bin":"released model"}
        inputs = {}
        for relative,content in {"scripts/phase6_native_baseline.py":"original native driver",
                                 "outputs/phase6/native_baseline/plan.json":"original plan",
                                 "scripts/phase6_fusion.py":"original learner driver",
                                 "outputs/phase6/fusion_v2/fit_complete.json":"original fit",**assets}.items():
            path = self.root/relative; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(content)
            inputs[relative] = native.digest(path)
        inputs[self.calpath.relative_to(self.root).as_posix()] = native.digest(self.calpath)
        self.asset_hashes = {p:inputs[p] for p in assets}
        self.lock = {"input_sha256":inputs,"learner_output":"outputs/phase6/fusion_v2"}
        self.write_json(self.lockpath,self.lock)
        self.write_json(self.binding,{"protocol_sha256":native.digest(self.lockpath),
            "source_manifest_sha256":native.digest(self.sources),"source_count":1800,"test_scoring_started":False})
        variables = [(guard,"ROOT",self.root),(guard,"WRAPPER",self.wrapper),(guard,"GUARD",self.out/"guard_locked.json"),
            (guard,"SOURCE_RECEIPT",self.binding),(guard,"EXPECTED_ROOT_SHA",native.digest(self.lockpath)),
            (guard,"EXPECTED_SOURCE_RECEIPT_SHA",native.digest(self.binding)),(native,"ROOT",self.root),
            (native,"OUT",self.out),(native,"SOURCES",self.sources),(native,"LOCK",self.lockpath),
            (native,"ASSET_HASHES",self.asset_hashes),(native.official,"__file__",str(previous))]
        for obj,key,value in variables: self.context.enter_context(patch.object(obj,key,value))
        self.context.enter_context(patch.object(native,"plan",return_value={"required_source_counts":required,"model_asset_sha256":self.asset_hashes}))

    @staticmethod
    def write_json(path,value): path.write_text(json.dumps(value),encoding="utf-8")

    @staticmethod
    def write_csv(path,rows):
        with path.open("w",encoding="utf-8",newline="") as handle:
            writer = csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    def test_seal_and_resume_bind_unchanged_original_functions(self):
        functions = {n:getattr(native,n) for n in ["load_official","infer","negative_cut","calibrate","paired_metrics","analyze"]}
        first = guard.seal()
        self.assertEqual(guard.validate_lock(),native.digest(self.lockpath))
        self.assertEqual(guard.seal(),first)
        self.assertIn("scripts/phase6_native_bound.py",first["input_sha256"])
        self.assertIn("scripts/native_baseline.py",first["input_sha256"])
        self.assertFalse(first["detector_preprocessing_scoring_cut_and_statistics_functions_changed"])
        for name,method in functions.items(): self.assertIs(getattr(native,name),method)

    def test_rejects_changed_wrapper_after_seal(self):
        guard.seal(); self.wrapper.write_text("changed wrapper")
        with self.assertRaisesRegex(ValueError,"guard dependencies changed"): guard.validate_lock()

    def test_rejects_changed_source_receipt(self):
        self.binding.write_text("changed receipt")
        with self.assertRaisesRegex(ValueError,"source receipt changed"): guard.seal()

    def test_rejects_changed_frozen_source_manifest(self):
        with self.sources.open("a") as handle: handle.write("changed\n")
        with self.assertRaisesRegex(ValueError,"source manifest changed"): guard.seal()

    def test_rejects_guard_added_after_native_inference_started(self):
        self.write_json(self.out/"calibration_started.json",{})
        with self.assertRaisesRegex(RuntimeError,"after native inference began"): guard.seal()

    def test_uses_current_learner_and_requires_main_threshold_lock(self):
        guard.seal()
        self.write_json(self.out/"calibration_predictions.csv",{})
        self.write_json(self.out/"thresholds_locked.json",{"root_protocol_sha256":native.digest(self.lockpath),
            "calibration_predictions_sha256":native.digest(self.out/"calibration_predictions.csv")})
        stale = self.root/"outputs/phase6/fusion"; stale.mkdir()
        self.write_json(stale/"thresholds_locked.json",{})
        with self.assertRaisesRegex(RuntimeError,"root-selected learner thresholds"): guard.validate_lock("test")

    def test_cannot_leave_project_for_a_dependency(self):
        with self.assertRaisesRegex(ValueError,"leaves the project"): guard.project_file("../outside")


if __name__ == "__main__": unittest.main()
