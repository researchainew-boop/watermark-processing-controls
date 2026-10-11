# Sources for the learned-watermark and diffusion extension

Research checked October 7, 2026. This note records the source review and
candidate choices made before execution. The October 9 extension has now
executed official TrustMark Q and cached Realistic Vision DDIM/VAE controls;
see extension_run.md for actual settings, saved status, and measured evidence.
Alternative methods below remain proposals, not additional experiments.

## Learned watermark: Adobe TrustMark

TrustMark is a suitable second, learned post hoc watermark. Its paper is T. Bui, S. Agarwal, and J. Collomosse, “TrustMark: Robust Watermarking and Watermark Removal for Arbitrary Resolution Images,” ICCV 2025, pp. 18629–18639. The publisher entry is [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Bui_TrustMark_Robust_Watermarking_and_Watermark_Removal_for_Arbitrary_Resolution_Images_ICCV_2025_paper.html).

The authors' [repository](https://github.com/adobe/trustmark) states that MIT licensing covers its code and downloaded model files. Preserve the license notice. [PyPI](https://pypi.org/project/trustmark/) currently lists version 0.9.2, released September 8, 2026. The [package configuration](https://raw.githubusercontent.com/adobe/trustmark/main/python/pyproject.toml) requires Python >=3.8.5, torch >=2.3.1, torchvision >=0.18.1, omegaconf, NumPy, lightning, six, and einops.

Prefer the official Python implementation on CPU if already compatible with the installed torch/torchvision environment. Instantiate variant Q with `device='cpu'` and `loadRemover=False`; the latter avoids loading an unnecessary remover. The [source](https://raw.githubusercontent.com/adobe/trustmark/main/python/trustmark/trustmark.py) implements a raw 100-bit option with ECC disabled as well as BCH modes. Record the choice explicitly; raw bit agreement is different from default BCH watermark verification. Variant C is described in that source as a smaller ResNet-18 decoder alternative, with a quality tradeoff.

The current default is BCH_5: the [data layer](https://raw.githubusercontent.com/adobe/trustmark/main/python/trustmark/datalayer.py) allocates **61 data bits**, 35 error-correction bits, and four schema bits. Query `schemaCapacity()` in the installed version rather than assuming a 57-bit message. If carrying the study's 16-bit payload, define the remaining data bits before scoring and compare the entire decoded expected message; successful ECC decoding alone is not a payload match. Report the resulting clean false-match rate separately from the classical baseline's 13-of-16 threshold.

The current official Python source identifies these Q downloads and MD5 values. Record SHA-256 and file size after transfer as well:

- `trustmark_Q.yaml`: `fe40df84a7feeebfceb7a7678d7e6ec6`.
- `decoder_Q.ckpt`: `4ced90e9cfe13e3295ad082887fe9187`.
- `encoder_Q.ckpt`: `700328b8754db934b2f6cb5e5185d81f`.

They use the same CDN listed below. The [model module](https://raw.githubusercontent.com/adobe/trustmark/main/python/trustmark/model.py) imports `torchmetrics.functional`, which must be available through the Lightning dependency. Avoid removing declared dependencies simply because encoding uses inference only.

Official encoder and decoder ONNX models also exist. The authors' [Rust implementation](https://github.com/adobe/trustmark/tree/main/rust) supports encoding and decoding; the JavaScript implementation supports decoding. The [model-fetch source](https://raw.githubusercontent.com/adobe/trustmark/main/rust/crates/xtask/src/main.rs) supplies the current CDN:

- `https://cai-watermark.adobe.net/watermarking/trustmark-models/encoder_Q.onnx`
- `https://cai-watermark.adobe.net/watermarking/trustmark-models/decoder_Q.onnx`

The inspected official pages do not publish reliable exact model byte sizes. Record transfer bytes, final size, and SHA-256 when actually downloading. A custom ONNX adapter must follow the selected official implementation consistently: Python Q decoding uses 245 pixels, whereas [Rust ONNX decoding](https://raw.githubusercontent.com/adobe/trustmark/main/rust/src/lib.rs) uses 256. Their residual postprocessing also differs. Do not silently call a modified adapter identical to the default Python pipeline.

## Actual diffusion reconstruction

Prefer the complete model already present in the local cache: `SG161222/Realistic_Vision_V5.1_noVAE`, revision `1e9f017a7b1eaefb63a1900ea6c5953d2739fd21`. Local inspection by the main agent found its full configuration, tokenizer, UNet, text encoder, and VAE. These are local findings rather than measurements published by the model author. The creator's current [model card](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE) specifies CreativeML OpenRAIL-M and places this family under Stable Diffusion 1.5. It recommends `stabilityai/sd-vae-ft-mse-original` for V5.1. Record the actual cached VAE's identity and hash, since the pipeline name alone does not establish which VAE was used. The inspected card does not provide complete training provenance, and its current text should not be presented as verified historical content of the cached revision.

Reuse this meaningful pretrained model before downloading another. If the cached pipeline is unusable, a fallback is [Segmind tiny-sd](https://huggingface.co/segmind/tiny-sd), a trained architectural distillation of Realistic Vision V4.0. Its card reports 125,000 training steps at 512-pixel resolution and CreativeML OpenRAIL-M licensing. It is distinct from random miniature pipelines used to test software. Cite its identity explicitly rather than describing it as the full original Stable Diffusion 1.5 model.

The publisher's [file listing](https://huggingface.co/segmind/tiny-sd/tree/main) totals approximately 1.06 GB: [UNet](https://huggingface.co/segmind/tiny-sd/tree/main/unet) 647 MB, [VAE](https://huggingface.co/segmind/tiny-sd/tree/main/vae) 167 MB, and [text encoder](https://huggingface.co/segmind/tiny-sd/tree/main/text_encoder) 246 MB, plus small tokenizer/configuration files. Capture the exact revision and downloaded hashes. Converting the half-precision weights to float32 approximately doubles weight RAM; this is an estimate, not a peak-memory measurement.

[Diffusers 0.35.1](https://pypi.org/project/diffusers/0.35.1/) is a documented stable pin, rather than an assertion that it is the newest release. Use transformers, accelerate, huggingface-hub, safetensors, and a compatible existing torch installation. Record the versions that actually pass import and inference checks. Its [image-to-image documentation](https://huggingface.co/docs/diffusers/v0.35.1/en/api/pipelines/stable_diffusion/img2img) describes latent noise followed by diffusion denoising, with reconstruction strength controlling noise and the number of effective steps. Empty prompt and guidance scale 1 avoid additional scene instructions. CPU float32 or CUDA float16 are candidates; do not use the model page's autogenerated bfloat16 example on a GTX 1050 Ti. Batch size one and [CPU offloading](https://huggingface.co/docs/diffusers/v0.35.1/en/optimization/memory) can reduce GPU memory at a runtime cost. Benchmark before committing to the sample size.

This is a meaningful diffusion reconstruction attack, but it is not an exact reproduction of Zhao's custom implementation. The authors' [WatermarkAttacker code](https://github.com/XuandongZhao/WatermarkAttacker) separately specifies latent noise steps and its own reconstruction pipeline. Keep a VAE-only round trip, image resizing, and actual diffusion denoising as separate conditions.

### Local runtime compatibility

The main agent observed a GTX 1050 Ti with 4 GB VRAM, driver 560.94, and a bundled Python 3.12 runtime. The official [PyTorch installation history](https://pytorch.org/get-started/previous-versions/) specifies the Windows pair `torch==2.5.1`, `torchvision==0.20.1` from the CUDA 12.1 wheel index. This satisfies TrustMark's declared minimum package versions. NVIDIA's [CUDA 12.1 release notes](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/index.html) specify Windows minimum driver 527.41, below the observed driver.

NVIDIA's [CUDA programming guide](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-c-programming-guide/index.html) states that a binary targeting compute capability 6.0 can run on 6.1. The PyTorch installation page does not enumerate that wheel's embedded architectures, so record `torch.cuda.get_arch_list()`, the device capability, and a completed GPU operation before reporting compatibility as measured. Do not infer it from `is_available()` alone. The executed diffusion extension uses batch one, fp16 UNet/text encoder, fp32 VAE, and CPU offload. Its completion marker records 288 outputs; protocol.json records exact settings and hashes, and manifest.csv records output timings. The model card's autogenerated bfloat16 example is unsuitable for this Pascal GPU.

## Content controls and a small external evaluation

The primary [Synthbuster release](https://zenodo.org/records/10066460) contains 1,000 outputs per generator for nine generators, loosely matched by prompts to RAISE-1k photographs. Its ZIP is 12.4 GB, and RAISE images are separate. This is an established content-matched candidate, but not the smallest download. The authors of a detection evaluation provide the [download script](https://raw.githubusercontent.com/grip-unina/ClipBased-SyntheticImageDetection/main/data/synthbuster_download.sh) and a RAISE-1k mirror; the script specifies noncommercial use and citation of both datasets.

A smaller proposed alternative is to select previously unused COCO photographs by a saved seed, then generate images from their existing COCO captions using the trained model above. The official caption archive is `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`. [TensorFlow Datasets' checksum registry](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/coco_captions.txt) records 252,907,541 bytes for that archive. Prefer a small fixed sample of complete caption/photo pairs, preserve the full prompt and generation settings, screen against all existing sources, and cluster uncertainty by originating scene. This is a proposed new benchmark slice, not an existing measured result.

Before scoring, fix the sample, attack strengths, seeds, image-quality criteria, and evaluation plan. Include clean, marked, resizing-only, VAE-only, and diffusion-denoised conditions with the same input/output sizing. Apply each setting to marked synthetic images and photographic inputs, then report bit recovery, fixed-threshold false matches, probe/fusion performance, PSNR/SSIM, and source-clustered uncertainty. A diffusion-reconstructed photograph contains generated pixels: call it a photographic-source reconstruction and define labels by input-source provenance. A positive score on it must not automatically be interpreted as a false accusation against an untouched photograph. Do not retune final thresholds on the new evaluation results.
