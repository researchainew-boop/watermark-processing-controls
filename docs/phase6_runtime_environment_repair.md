# Secondary attack environment repair

The first secondary-attack attempt completed calibration PGD and factorized
reconstruction images, then stopped before SD1.5 inference because Diffusers
0.35.1 rejected PEFT 0.13.2. No parameters, model bytes, source choices, attack
settings, primary thresholds or scientific execution scripts were changed.

The resumed environment puts the existing `.tools/lcm-deps` directory before
`.tools/extension-deps`. It imports PEFT 0.17.0 from that recorded directory,
while retaining Torch 2.5.1+cu121, NumPy 1.26.4, Diffusers 0.35.1 and Transformers
4.46.3. The cached calibration PGD/reconstruction images are hash-checked before
reuse. This is an additive environment note; the original frozen protocol and
score receipts are preserved.

An interrupted continuation was resumed from its recorded manifests. Completed
calibration features and separate secondary cuts are verified before reuse.
Partially completed test attack images are checked against their recorded hashes.
The same registered attack and source/payload seeds are retained.

An unchanged calibration continuation reserialized the already completed CSV;
default floating-point CSV parsing changed bytes without changing image inputs.
The exact earlier manifest was restored by recomputing pixel quality/budget
metadata and replaying its known earlier serialization stage. Restoration was
accepted only when the resulting full SHA-256 equaled the original feature
receipt. The reserialized version remains archived. The supplemental controller
now skips completed derivation and validates its checksum before reuse.

Interrupted test PGD calls also left files beyond their ten-cover manifest
checkpoint. GPU PGD retries need not produce identical bytes despite the fixed
random seed. Only unrecorded files were moved into an internal interrupted-run
archive before retry; committed inputs were hash-checked and retained. Test
feature inference had not started. No retry was selected by detector/decoder
performance, and all registered sources/settings remained fixed.

This environment repair does not establish scientific novelty or independent
reproduction. The secondary outcomes must remain separate from the original
four-view primary endpoints.
