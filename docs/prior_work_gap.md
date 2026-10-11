# Contribution and Related-Work Wording

The the conference submission paper evaluates expected-payload watermark recovery, frozen
origin probes, and fusion under matched transforms of real and synthetic
images. Its defensible contribution is the measured role of controls and
observed failure cases. It does not introduce a new backbone, cryptographic
watermark, intent detector, or universally better origin score.

## Positioning

- CIFAKE, universal fake-image detection, and GenImage concern detection and
  evaluation. Avoid saying all prior detectors omit transformed real controls
  without reviewing their exact protocols.
- Invisible-watermark removal and UnMarker motivate attack scope. The study's
  legacy grid shift/JPEG/noise transform is not diffusion reconstruction.
  The October 9 extension separately executes actual DDIM reconstruction
  and a VAE-only control. It is not an exact implementation of Zhao's or
  UnMarker's published attack.
- Acknowledge previous combinations of provenance and forensic evidence.
  Describe the particular controls measured here without claiming the
  generate/edit/detect sequence is unprecedented.
- Distinguish payload recovery from authenticity. One fixed expected payload
  in the corrected classical core does not establish authorship or resist
  an attacker who knows the bits. The extension's three payloads and second
  watermark do not establish cryptographic authentication either.

Check paper-specific statements against primary text. Avoid oversimplifying
whole papers as "detection only" or "removal only."

## Claims supported by the experiment

The study uses the same listed transforms on real and watermarked AI images,
compares passive origin probes and fusion, and evaluates removal of matched
real controls from training. Some operations preserve the payload; others
make fusion worse. Quote core measurements and clustered intervals from
outputs/conference_clean/, and identify extension results separately using
outputs/extension/ and outputs/extension_controls/. The extension compares
classical >=13/16 verification with TrustMark Q exact 61-bit expected-message,
schema, and ECC verification; do not treat these criteria as equivalent.
Its completed diffusion benchmark uses one checkpoint and fixed settings.

The secondary S-A classifier is exploratory. Its positive class records a
watermarked AI image after an operation, including cases where the payload
survives. This does not verify humanification or infer intent.

## Claims to avoid

- An unremovable watermark or authenticated secret-key provenance system.
- Universal detection inferred from ordered collection prefixes.
- Diffusion regeneration attributed to the legacy grid-shift operation,
  or an exact Zhao/UnMarker reproduction attributed to the new DDIM run.
- Humanification or editing intent inferred from the S-A label.
- Elimination of collection bias by resizing or duplicate screening.
- A preregistered confirmation study: this correction is retrospective.
- Guaranteed acceptance or a required minimum of 100,000 images.

Historical CIFAKE and software demonstration results are not conference
measurements.

