"""Cheap interface checks for the authorized Phase6 attack evaluation.

Published-model parity and quality anchors are separately recorded by the
development driver. These checks do not allocate GPU memory or run models.
"""
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from phase6_attacks import digest, residual_budget, source_seed, validate_image


def fixture_pixels():
    rng = np.random.default_rng(311)
    reference = rng.integers(30, 226, size=(128, 128, 3), dtype=np.uint8)
    delta = rng.integers(-20, 21, size=reference.shape)
    candidate = np.clip(reference.astype(np.int16)+delta, 0, 255).astype(np.uint8)
    return reference, candidate


class AttackInterfaceChecks(unittest.TestCase):
    def test_source_seed_is_reproducible_and_arm_independent(self):
        self.assertEqual(source_seed("same_source", 0), source_seed("same_source", 0))
        self.assertNotEqual(source_seed("same_source", 0), source_seed("same_source", 1))
        self.assertNotEqual(source_seed("same_source", 0), source_seed("other_source", 0))
        self.assertTrue(0 <= source_seed("same_source") < 2**32)

    def test_fixed_quality_budget_reaches_target_without_amplifying_attack(self):
        reference, candidate = fixture_pixels()
        result, meta = residual_budget(reference, candidate, 35)
        self.assertEqual(result.shape, reference.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(0 <= meta["gain"] <= 1)
        self.assertTrue(meta["target_reachable"] and meta["within_tolerance"])
        self.assertLessEqual(abs(cv2.PSNR(reference, result)-35), .1)
        self.assertTrue(np.all(np.abs(result.astype(int)-reference.astype(int)) <= np.abs(candidate.astype(int)-reference.astype(int))))
        repeated, repeated_meta = residual_budget(reference, candidate, 35)
        self.assertTrue(np.array_equal(result, repeated))
        self.assertEqual(meta, repeated_meta)

    def test_unreachable_quality_is_disclosed_without_strengthening(self):
        reference, _ = fixture_pixels()
        candidate = reference.copy()
        candidate[0, 0, 0] += 1
        result, meta = residual_budget(reference, candidate, 35)
        self.assertTrue(np.array_equal(result, candidate))
        self.assertFalse(meta["target_reachable"])
        self.assertEqual(meta["gain"], 1)

    def test_input_shape_and_type_checked(self):
        reference, candidate = fixture_pixels()
        with self.assertRaises(ValueError):
            validate_image(reference.astype(np.float32))
        with self.assertRaises(ValueError):
            validate_image(reference[:, :, 0])
        with self.assertRaises(ValueError):
            residual_budget(reference, candidate[:64], 35)

    def test_sha256_records_exact_file_bytes(self):
        scratch = Path(__file__).resolve().parents[1] / "outputs/phase6/attack_development/checks"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            if not Path(folder).resolve().is_relative_to(scratch.resolve()):
                raise ValueError("Check scratch path escaped development directory")
            path = Path(folder) / "bytes.bin"
            path.write_bytes(b"abc")
            self.assertEqual(digest(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


if __name__ == "__main__":
    unittest.main()
