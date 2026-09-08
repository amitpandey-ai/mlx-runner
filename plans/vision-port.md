# Vision port (Qwen3.8-27B native VLM) — plan

## Context

Text engine done (native GDN + MTP + bench, no Python). The checkpoint ships
a full vision tower (333 `vision_tower.*` BF16 tensors, currently dropped at
load) and the model card leads with VL chops (OSWorld-Verified 84.3,
MathVision 94.6 w/ CI). Model card: https://huggingface.co/Qwen/Qwen3.8-27B.
Input format (OpenAI-style): `content: [{type: image_url, image_url: {url}},
{type: text, ...}]`, plus video (`fps: 2.0`, 4–768 frames). Thinking mode
(`<think>` 248068/69) is orthogonal and stays as-is.

## Approach

1. **Image/video load + preprocess, `src/vision_preproc.zig` (hermetic).**
   - Decode JPEG/PNG (mlx-infer has `jpeg.zig` to crib; add PNG via system
     ImageIO? prefer stb-style vendored decoder — decide at implementation,
     no new system deps), RGB convert, resize honoring min/max pixels
     (65536..16777216 image; video 4096..25165824, 4..768 frames @2fps),
     rescale 1/255, normalize mean/std 0.5, patchify (16px spatial,
     temporal_patch_size 2), 2×2 spatial merge. Pure-CPU, file-local tests
     with a tiny fixture image (shape math + mean/std spot values).
2. **ViT tower, `src/vision.zig` (needs step 1 weights only).**
   - 27 blocks: fused qkv `[3456,1152]` + proj, both WITH bias (unlike text);
     norms WITH bias; MLP 4304 with `gelu_pytorch_tanh`; learned pos_embed
     `[2304,1152]`; patch_embed 3D conv. Merger: norm+bias + 2 linears+bias
     to 5120. All bf16, batch-1, no cache (full-sequence every prefill).
   - Loader: stop dropping `vision_tower.*` (keep `--no-vision` as a real
     RAM-saving off-switch, default ON when tower present).
   - Tests: block shape/dtype/no-NaN on random small inputs (linked suite).
3. **Splice + template (needs step 2 + engine).**
   - `Model.forwardEmbeddings` path: text `input_embeddings` with vision
     rows spliced at `<|image_pad|>` (248056) / `<|video_pad|>` (248057)
     positions, framed by `<|vision_start|>` (248053) .. `<|vision_end|>`.
   - `renderChat` gains the vision branch: download-or-file URLs?
     Decide: local files + http(s) via builtin fetch (no new deps).
     Non-decodable input -> `error.VisionUnsupported` (keep the error).
   - MTP: vision participates in prefill only (drafts stay text-only);
     MTP cache seeding uses the spliced hidden states unchanged.
4. **Cut over + verify.**
   - `--no-vision` becomes meaningful; help/version updated.
   - VQA probe: model-card math diagram URL + 2 local images, greedy
     temp-0 answers judged manually (no bit oracle exists); text bench
     re-run to prove no regression (agreement gate + acceptance band).

## Verification

- `zig build test` + `test-mlx` green (preproc shapes, ViT block invariants).
- VQA probe answers are sensible (diagram factors, doc OCR spot-check).
- `--bench` numbers within noise of text-only (vision weights add RAM
  only when enabled; text path untouched).
- `grep -ri "python" src/ build.zig` still clean (preproc stays native).

## Assumptions & contingencies

- bf16 ViT throughout; no quant in this phase.
- Video = frames-as-images + temporal patchify (no motion model beyond the
  tower's temporal conv); hour-scale video is a context-window concern
  (262144 ctx ≈ frames*tokens budget enforced, overflow -> error).
- If ImageIO decode proves cleaner than vendored decoders, use it (macOS
  system framework, no install).
