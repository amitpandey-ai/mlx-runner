# mlx-runner — project notes (auto-included)

Native Zig 0.17 inference for Qwen3.8-27B (qwen3_5 GDN hybrid, bf16). No
Python in the loop. Pinned: Zig 0.17.0-dev.1818+7051f8e73, mlx==0.32.2.

## Build / toolchain

- `bash scripts/fetch-zig.sh` → `.zig-toolchain/zig`; `bash scripts/build-mlx.sh`
  → `lib/mlx` (libmlx.dylib, libmlxc.dylib, mlx.metallib, headers).
- Since mlx 0.32 the PyPI `mlx` wheel is a thin shim; native bits live in
  `mlx-metal`. PyPI wheels are built WITHOUT NAX kernels — prefer the local
  NAX-enabled 0.32.2 when present (script asserts version + NAX kernel
  strings), else wheel fallback. Building NAX from source needs Xcode 26.2+
  with accepted license (`sudo xcodebuild -license`, `sudo xcode-select -s`).
- `find_package(MLX)` MUST resolve the staged tree: stage `share/cmake` +
  `lib/cmake` too, or cmake silently picks brew mlx (header skew breaks the
  mlx-c build).
- mlx-c has no `freqs=null` literal: pass `mlx_array{ .ctx = null }`.
  SDPA mask off = `""` (mask_mode must never be NULL). Rope partial dims are
  handled inside `mlx_fast_rope` (dims=64 on full 256-dim heads, no reshape).
- `mlx_load_safetensors` MUST use the CPU stream: GPU-stream loads fail at
  eval with `[Load::eval_gpu] Not implemented`.
- Never read >2GB in one pread (INVAL): shard headers are read as 8+n bytes
  only, never whole multi-GB files.

## Zig 0.17 snapshot idioms (this toolchain)

- HashMap managed (`init/deinit/put/get`), ArrayList unmanaged
  (`.empty`, allocator per op). No `Allocator.dupeZ`.
- `takeDelimiter('\n')` (`?slice`, null at EOF) for REPL stdin —
  `takeDelimiterExclusive` returns empty instead of blocking on pipes.
- Extern decls without lib calls link fine; any LIVE mlx call needs libmlx:
  keep mlx-touching tests in `mlx_test.zig` (linked), pure tests in
  `tests.zig`. `refAllDecls` in mlx_test forces analysis so imported files'
  tests join the linked runner.
- Timing: `std.Io.Clock.Timestamp.now(io, .awake)` +
  `durationTo(...).raw.toMilliseconds()` (no `std.time.Timer`,
  no `std.posix.clock_gettime/getrandom`; entropy via `std.c.arc4random_buf`).
- `zig test` discovers tests transitively through imports.

## Model facts (Qwen3.8-27B, `~/opt/models/mlx_models/Qwen3.8-27B`)

- 64 layers (interval 4: idx 0,1,2 GDN, 3 full, …), hidden 5120, 24/4 heads,
  head_dim 256, rope 64 @ 1e7 partial, intermediate 17408, vocab 248320,
  bf16-only, 1199 tensors (333 vision dropped → 866 loaded).
- GDN dims: Hk=16/Hv=48, Dk=Dv=128, conv [10240,4,1] groups=10240,
  qkv split [2048,2048,6144], fold-ins q·Dk⁻¹/k·Dk⁻⁰·⁵, g/beta f32, state f32.
- Full attn: q_proj 2×(24×256) split queries+gate, per-head q/k RMSNorm,
  native GQA in SDPA (never repeat K/V), `o_proj(sigmoid(gate)·out)`.
- MTP: 15 tensors, fc [5120,10240], 1 full-attn layer, shared lm_head;
  consumes backbone PRE-norm hidden (fork's `return_hidden` is pre-norm).
- Tokenizer: BPE 248044 + 33 added (248044..248076), GPT-2 split + ByteLevel,
  NFC normalizer SKIPPED (documented; only decomposed input diverges),
  prompts encoded as-is, EOS [248046, 248044].
- Sampling (fork-exact): filters on UNSCALED-normalized basis, temp at
  categorical + accept-LPs; top_p==0 means OFF; greedy = argmax.
- Greedy MTP-vs-base is NOT bit-identical in hardware fp (batched Lq=2 vs
  incremental Lq=1 kernel paths + accepted-state drift → ~0.5% flips for bf16, 15-40% for quant due to qmm noise):
  bench gates acceptance ≥35% + agreement ≥98% bf16 / ≥80% quant, not bytes.

## Gotchas fixed (do not regress)

- `Model.wm` must be OWNED by value: a borrowed pointer to the loader frame
  dangles (manifested as phantom MissingWeight).
- GDN snapshot for MTP rollback is mid-loop at n_confirmed (conv slice +
  retained ssm), not pre/post-forward.
- MTP KV trims on reject (deviation from fork, keeps positions aligned).
- fast.rope `traditional=false` = half-interleaved pairs (verified vs mx).
- `mlx_array_new_data` copies synchronously; lazy graphs keep nodes alive
  past Zig-handle frees (safe), but eval before reading back.
- KV pre-alloc: `Caches.reserveKv` + `MtpState.kv.reserve` for `prompt+max_tokens` before prefill avoids step-256 reallocs (peak -2 GB); `mlx_clear_cache` once after prefill loop, not per chunk.

## Serve API (`--serve`, OpenAI + Anthropic)

- `src/http_api.zig` = pure protocol (normalize/builders/SSE/stops), no
  engine import → hermetic tests. `src/server.zig` = sockets + dispatch +
  runGen glue (needs libmlx; verified live, not in hermetic suite).
- `refAllDecls(server.zig)` in tests.zig BREAKS the hermetic link (drags the
  engine call graph in). Never re-add it; server is covered by live curl.
- Usage = `last_stats` engine counts, NOT re-encoded text (BPE merges runs
  like "!!!!" on re-encode and undercounts). Hot-cache hits report the
  previous call's stats (documented approximation; text always exact).
- SSE is buffered (monolithic generate → UTF-8-safe 64B chunks after the
  fact). Sequential connections; `Writer.print` format strings: validate
  brace parity with the python checker (bitten twice on `}}}` runs).
- `std.Io.net`: `addr.listen(io,.{})` → accept → `stream.reader/writer`
  return structs with `.interface` for `http.Server.init`; body via
  `readerExpectNone` vs `std.Io.Reader.ending`; `Clock.real.now(io)` = epoch.

## Multi-model + quant serving

- Supported: qwen3_5 bf16 / mxfp8 / nvfp4. Affine (`-8-bit`, BF16 scales)
  rejected at load (`AffineUnsupported`) per user scope — never add it back
  without asking.
- Quant spec is per CHECKPOINT (`WeightMap.quant`), inferred from shapes:
  U8 scales + packed/scales ratio r → r=8 mxfp8(32,8), r=2 nvfp4(16,4).
  Verified against mlx's own `_defaults_for_mode` (affine 64, mxfp8 32/8).
- Forward: `namedLinearQ`/`layerLinearQ`/`mlpForwardQ` (model.zig) +
  `mtpLinearQ` (mtp.zig); dense fallback iff scales absent AND weight is not
  U32 (mxfp8 lm_head). qmm args mirror `nn.QuantizedLinear.as_linear`:
  transpose=true, null biases, per-map group/bits/mode.
- Proven by linked synthetic test (MLX quantize → qmm ≈ CPU matmul over
  MLX dequantize, tol 0.05) + live coherent 40-token nvfp4 output.
- Registry (server.zig): `--model` repeatable, ids = dir basenames
  (duplicates rejected at startup), lazy load, LRU evict past
  `--max-resident-models` (default 1). Requests route via `model` field.
- Per-model config (`src/config.zig:loadFile` + `src/server.zig:ModelSpec`):
  `--config file.json` gives per-model `alias` (API id), `sampling`
  (temp/top_p/top_k/min_p/seed/max_tokens), `config` (ctx_size/mtp/mtp_gamma).
  Priority `request > CLI > per-model config > generation_config.json`.
  `max_tokens` default = request else per-model else 1024 (serve) / 256 (CLI).
  `ctx_size` 0 = model max (`max_position_embeddings`, 262144, assume RAM).
  `GET /v1/models` lists aliases. `FileConfig` dupes host/cache strings (parsed
  JSON freed). `ModelSpec.max_tokens` plumbed via `Resolved` into routes.
- Hermitian split: `refAllDecls(server.zig)` in tests.zig BREAKS the
  hermetic link (engine call graph) — server covered by live curl only.
  `config.zig` is hermetic (refAllDecls in tests.zig, 33 tests).

## Fused GDN kernel (prefill 2-3.7x)

- `src/gdn_packed.metal` = mlx_lm packed kernel verbatim except scalar `T`
  input (mlx-c takes arrays only): whole-word `T` -> `(T[0])`, T passed as
  uint32[1]. Verified: only InT/StT identifiers contain T otherwise.
- Call in model.zig (`gdnKernelRun`): grid (32, Dv/8, B*Hv), tg (32,2,1),
  templates InT=y dtype, StT=f32, Dk/Dv/Hk/Hv from cfg. Handle compiled once
  (global cache). `Model.fused_gdn` (default true) is the hatch; eligibility
  mirrors reference `packed_eligible` (dk 128, dv%8, g ndim3 f32, st f32).
- CRITICAL layout rule: kernel takes PRE-REPEAT qsc/ksc [B,T,Hk,Dk] and fans
  out to Hv itself (hk_idx). Passing qr/kr gives fluent garbage ("rooms of
  rooms") — caught by greedy agreement, never by shapes.
- MTP verify (n_confirmed>0) splits into two kernel calls (prefix + rest)
  with the conv/mid-state snapshot between, mirroring the ops loop.
- Proof stack: linked synthetic test (CPU recurrence vs kernel, y 0.05 /
  state 1e-3) + fused-vs-ops byte-identical + MTP-vs-base byte-identical.
- Measured (32K ctx bench, ReleaseFast, prefill 2048 chunk): prefill bf16 700/664, mxfp8 511/505, nvfp4 676/640; decode MTP bf16 17.8, mxfp8 23.9, nvfp4 39-40 tok/s (base 8.5/15.2/26.9). Fused gives 2-3.7× prefill; quant gives 3× decode over bf16.
- Decode is at the memory roof (~400 GB/s); only speculative decoding moves it. MTP γ=2 (~1.3× more) implemented behind `--mtp-gamma 2` (draft 2 → verify 3, deeper rollback), default γ=1 stable; quant agreement is lower (nvfp4 81%, mxfp8 58% vs bf16 99.9% greedy) so bench now gates 80% for quant, 98% for bf16.

## True APC (prefix KV cache)

- `src/apc.zig:ApcCache` — token-id longest-prefix LRU (32 entries, 2GB) cloning `Caches.kv/gdn` + `MtpState.kv` via `mlx_array_set` retain. `findLongestPrefix` scans ≤32 entries, `e.tokens ⊑ prompt_ids`. Hit restores GDN `conv/ssm` + KV `keys/values/views` + `len/cap` and `MTP KV`, clears pending, sets `pos=cached_len` so remaining prefill is `prompt_len - cached_len`. Verified miss/hit with nvfp4: 510 tok prefill 663ms → hit 509 cached +7 computed in 59ms (~11× prefill saving), greedy outputs byte-identical to no-cache.
- `src/model.zig:KVCache.clone/restoreFrom`, `GDNCache.clone/restoreFrom` — retain via `mlx_array_set`, snaps not cached. `src/engine.zig:generateUncached` does APC lookup after `tokenizer.encode`, `reserveKv` after restore, `apc.put` after prefill (clones full prompt state `prompt_len-1`). `GenerateStats.cached_tokens` holds saved count, `prefill_tokens` is full length, `prefill_ms` is actual compute. `bypass_cache` (bench) disables APC. Hot `PrefixCache` still handles exact completion hits; APC handles prefix.
- Invariant: BPE prefix stability — byte prefix that is not token-prefix (e.g., trailing space `"... "` + `"What"` → token 220 vs 3437) correctly misses; test prompts must end at token boundary (strip trailing space) for hit. History-growth prompts (prompt is extension of prior prompt) always hit.
- `src/mlx_test.zig:refAllDecls(apc.zig)` — APC tests run in linked suite; hermetic suite stays link-free.

## APC disk tier (quota + location)

- `src/apc.zig:ApcDisk` wraps `kv_checkpoint.KVStore` (`KVC\x01`, 48B header, SHA1, budget 4096 MiB) but payload is safetensors bytes of KV/GDN/MTP. `serializeToBytes` builds `map_string_to_array` (`kv.*.keys/values`, `gdn.*.conv/ssm`, `mtp.keys/values`, `tokens`, `cached_len`) + `map_string_to_string` metadata and `mlx_save_safetensors` to temp file → bytes; `deserializeFromBytes` writes bytes to temp file → `mlx_load_safetensors` (CPU stream) and restores `Caches` (`len/cap` from `cached_len` + shape, `kview/vview` via `mlx_slice`) + `MTP KV`. `store` → `KVStore.store(text, tokens.len, quant, model_id, ctx, payload)`, `load` → `KVStore.load` longest byte-prefix hit → deserialize → `cached_len`.
- `src/engine.zig:Engine.apc_disk` — hot miss → try `apc_disk.load` before full `caches.reset()`; after prefill `apc_disk.store` (same text/tokens/cached_len) in addition to `apc.put`. `bypass_cache` disables both. `hits/stores` logged as `[apc-disk] hit …`.
- `src/engine.zig:EngineConfig.apc_disk_quota` / `apc_disk_dir` + `src/main.zig` flags `--apc-disk` (`--apc-cache-disk`, `--apc-disk-quota`) and `--apc-disk-dir` (`--apc-cache-dir`). `parseSize` quota → MB, dir default `~/.mlx-runner/apc-disk/<hash>` (hash of model), or explicit absolute path. `apc_disk_enabled` if quota not `off` or dir set (default 10GB if dir without quota). Logged as `apc-disk 100MB (/tmp/…)`.
- Quota enforced by `KVStore.evictIfNeeded` (score `(hits+1)*tokens/size` with 6h half-life) both on `store` and on `init` scan. Verified nvfp4: 680 tok 342M file, 1GB quota holds 2 entries, 500MB after restart evicts to 1 entry, disk hit after restart gives 0-compute prefill (680 cached, 0 computed in 67ms), prefix hit (new question) finds longest byte-prefix (system alone) 679 cached +8 → 551ms vs 919ms full.

## Install script (latest release, macOS arm64)

- `install.sh` (root) + `scripts/install.sh` shim — one-liner `curl -fsSL https://raw.githubusercontent.com/amitpandey-ai/mlx-runner/main/install.sh | bash` installs `mlx-runner-macos-arm64.{tar.gz,zip}` from `release.yml` assets. Resolves `v*` via `api.github.com/repos/.../releases/latest` (`GITHUB_TOKEN` optional) fallback to `.../releases/latest` redirect. Verifies `checksums.txt` (`shasum -a 256`). Extracts `mlx-runner-macos-arm64/mlx-runner + lib/{libmlx,libmlxc,libjaccl,mlx.metallib}`.
- Installs to `PREFIX/bin/mlx-runner` + `PREFIX/lib/mlx-runner` (default `~/.local`, matches `Makefile:PREFIX/BINDIR/LIBDIR`). Flags: `--prefix/--bindir/--libdir`, `--version/--tag`, `--force`, `--dry-run` (no network), `--verbose`, `--uninstall`. Preflight: `Darwin` + `arm64` only, `sw_vers` warn <26.2, check `curl/tar/ditto|unzip/install_name_tool/otool`.
- Rpath fix (installed layout `bin -> ../lib/mlx-runner`): normalizes release-hardcoded `@executable_path/lib/libmlxc -> @rpath/libmlxc`, adds `@executable_path/../lib/mlx-runner` + `@executable_path/lib` to binary, adds `@loader_path` + change `@rpath/libmlx -> @loader_path/libmlx` to `libmlxc`, adds `@loader_path` to `libmlx`. Custom `BINDIR/LIBDIR` computes relative via `python3` `os.path.relpath`.
- Verified `v0.1.0` to `/tmp/prefix`: download 62M, checksum `b9e77c…`, staged fix, `otool -L` shows `@rpath/libmlxc`, `LC_RPATH @executable_path/../lib/mlx-runner`, `--metal-check` `metal_is_available=true`. `README.md:Installation` now one-liner first, source build second.
