# mlx-runner — Zig 0.17, no Python

MLX-native Metal LLM engine: pure-Zig inference for **Qwen3.8-27B** (Qwen3.5 GDN hybrid, 64 layers, bf16, 262k ctx) straight through `mlx-c`, including its native MTP speculative head and a base-vs-MTP benchmark. No Python interpreter, venv, or `uv` in the loop. NAX-enabled `libmlx` when available (M5, ~3.7× bf16 GEMM measured vs wheel).

> **Models are local only.** No Hugging Face downloader. Point `--model` to a local checkpoint directory containing `config.json`, `tokenizer.json`, and `model.safetensors.index.json`.

## Features

- **Qwen3.5 GDN hybrid** (64 layers, interval 4: 3× GDN + 1× full attention), bf16, 262k context, vocab 248k
- **Packed GDN kernel** (`src/gdn_packed.metal`, 2–3.7× prefill)
- **Quantized inference** — bf16 / mxfp8 / nvfp4 via `mlx_quantized_matmul` (per-checkpoint group/bits/mode inferred from shapes)
- **MTP speculative decoding** (γ=1 default, γ=2 draft-ahead, native fork parity)
- **True APC** — token-prefix KV reuse (hot LRU + SSD tier with quota/location, 11× prefill saving measured)
- **OpenAI + Anthropic HTTP API** (SSE, multi-model, lazy load + LRU evict)
- **No Python** — Zig 0.17, `mlx-c` FFI, Metal

## Requirements

- macOS 26.2+ with Xcode 26.2+ (Metal, NAX). Accept license: `sudo xcodebuild -license`, `sudo xcode-select -s`
- `cmake`, `webp` (`brew bundle install --file Brewfile` or `brew install cmake webp`)
- Zig `0.17.0-dev.1818+7051f8e73` (pinned; staged via `scripts/fetch-zig.sh`)
- `mlx==0.32.2` with `mlx-metal` (staged via `scripts/build-mlx.sh`; NAX if available, else wheel fallback)

## Installation

**One-liner (recommended) — latest prebuilt for macOS arm64:**

```sh
curl -fsSL https://raw.githubusercontent.com/amitpandey-ai/mlx-runner/main/install.sh | bash
mlx-runner --version
mlx-runner --metal-check
```

Options:

```sh
# Custom prefix (/usr/local needs sudo mkdir/chown first)
curl -fsSL https://raw.githubusercontent.com/amitpandey-ai/mlx-runner/main/install.sh | bash -s -- --prefix /usr/local --verbose

# Specific version
curl -fsSL https://raw.githubusercontent.com/amitpandey-ai/mlx-runner/main/install.sh | bash -s -- --version v0.1.0

# Uninstall
curl -fsSL https://raw.githubusercontent.com/amitpandey-ai/mlx-runner/main/install.sh | bash -s -- --uninstall
# or: ./install.sh --uninstall --prefix $HOME/.local

# Local run without curl
./install.sh --help
./install.sh --dry-run --verbose      # preview
./install.sh --prefix $HOME/.local    # → $HOME/.local/bin/mlx-runner + $HOME/.local/lib/mlx-runner
```

The script installs `mlx-runner-macos-arm64.{tar.gz,zip}` from the latest GitHub Release (`checksums.txt` verified via `shasum -a 256`), fixes `rpath`s for `$PREFIX/lib/mlx-runner`, and verifies with `--version` + `--metal-check`. Requires macOS on Apple Silicon (`arm64`); Intel or non-macOS exits with build-from-source hint. If `api.github.com` is rate-limited, set `GITHUB_TOKEN` or pass `--version`.

**Build from source (alternative):**

```sh
# 1) Zig toolchain (pinned)
bash scripts/fetch-zig.sh          # → .zig-toolchain/zig

# 2) MLX native libs (libmlx.dylib, libmlxc.dylib, mlx.metallib, headers)
bash scripts/build-mlx.sh          # → lib/mlx

# 3) Build mlx-runner (ReleaseFast)
.zig-toolchain/zig build -Doptimize=ReleaseFast
./zig-out/bin/mlx-runner --version

# Optional system install
make install PREFIX=$HOME/.local   # → $HOME/.local/bin/mlx-runner + $HOME/.local/lib/mlx-runner
```

`find_package(MLX)` must resolve the staged tree (`lib/mlx/share/cmake` + `lib/mlx/lib/cmake`) — otherwise cmake silently picks brew `mlx` and header skew breaks the `mlx-c` build.

## Model Setup

Place Qwen3.8-27B checkpoints locally (no downloader). Each checkpoint is a directory with at least:

```
config.json
tokenizer.json
model.safetensors.index.json
model-*.safetensors
generation_config.json
```

Supported checkpoints (qwen3_5 family):

| Variant | Quant | Packed | Scales | Notes |
|---------|-------|--------|--------|-------|
| `Qwen3.8-27B` | bf16 | — | — | 866 tensors (1199 total, 333 vision dropped) |
| `Qwen3.8-27B-mxfp8` | mxfp8 | U32 | U8 | r=8 (32,8) |
| `Qwen3.8-27B-nvfp4` | nvfp4 | U32 | U8 | r=2 (16,4) |

Affine (`-8-bit`, BF16 scales) is rejected at load (`AffineUnsupported`) — use bf16/mxfp8/nvfp4.

Example layout:

```sh
~/opt/models/mlx_models/Qwen3.8-27B        # bf16
~/opt/models/mlx_models/Qwen3.8-27B-mxfp8  # mxfp8
~/opt/models/mlx_models/Qwen3.8-27B-nvfp4  # nvfp4
```

Set a default for the CLI (no hardcoded user path):

```sh
export MLX_RUNNER_MODEL=~/opt/models/mlx_models/Qwen3.8-27B
# or pass --model explicitly
```

## Quick Start

```sh
MODEL=~/opt/models/mlx_models/Qwen3.8-27B

# Single prompt (greedy)
./zig-out/bin/mlx-runner --model $MODEL --prompt "The capital of France is" --temp 0 --max-tokens 10
# → Paris. ...

# MTP speculative decoding
./zig-out/bin/mlx-runner --model $MODEL --prompt "Hello" --mtp --max-tokens 32

# Sampling
./zig-out/bin/mlx-runner --model $MODEL --prompt "Hello" \
  --temp 0.7 --top-p 0.95 --top-k 20 --min-p 0.05 --seed 42 --max-tokens 32

# Chat REPL
./zig-out/bin/mlx-runner --model $MODEL --chat

# Benchmark (32k prefill + 2×400 decode, base vs MTP, temp 0, byte-equality gated)
./zig-out/bin/mlx-runner --model $MODEL --bench --bench-out bench.json
```

Sampling priority: `request > CLI > generation_config.json > hardcoded` (1.0/1.0/0/0). File defaults: `temp 1.0, top_p 0.95, top_k 20`. `--kv-quant` accepts only `off` (bf16-only build).

`--no-vision` is an accepted no-op (text-only engine; vision weights dropped at load; image/video chat content → `VisionUnsupported`).

## Server (OpenAI + Anthropic)

```sh
./zig-out/bin/mlx-runner --serve --port 11234 \
  --model ~/opt/models/mlx_models/Qwen3.8-27B \
  --model ~/opt/models/mlx_models/Qwen3.8-27B-mxfp8 \
  --model ~/opt/models/mlx_models/Qwen3.8-27B-nvfp4

curl http://localhost:11234/v1/models
curl http://localhost:11234/v1/chat/completions -d '{
  "model": "Qwen3.8-27B-nvfp4",
  "messages": [{"role":"user","content":"Hi"}],
  "max_tokens": 64
}'
curl http://localhost:11234/v1/completions -d '{
  "model": "Qwen3.8-27B",
  "prompt": "The capital of France is",
  "max_tokens": 10,
  "temperature": 0
}'
curl http://localhost:11234/v1/messages -d '{
  "model": "Qwen3.8-27B-mxfp8",
  "max_tokens": 64,
  "messages": [{"role":"user","content":"Hi"}]
}'
```

- `--model` is repeatable; request `model` ids are directory basenames (`Qwen3.8-27B`, `Qwen3.8-27B-nvfp4`, …) unless `--config` gives an `alias`. Absent `model` → first.
- Engines lazy-load on first request and LRU-evict past `--max-resident-models` (default 1; a 27B bf16 is ~55 GB, quants ~30 GB).
- Routes: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (`stream:true` → SSE + `[DONE]`), `POST /v1/completions`, `POST /v1/messages` (Anthropic, `stream:true` → `message_start/delta/stop`).
- Request sampling (`temperature`/`top_p`/`top_k`/`seed`, `max_tokens`, `stop`/`stop_sequences`) overlays per-model / CLI base. Priority: `request > CLI > per-model config > generation_config.json > hardcoded`.
- `ctx-size`: `0` = model max (`max_position_embeddings`, 262144 for Qwen3.8, extensible to 1M). Engine-global per model, not per-request; oversize prompt → `ContextOverflow` → 400. Assume enough unified memory for full context + weights.
- Image/tool content → 400 (text-only); `n>1` → 400; sequential requests (one in-flight).
- `stream:true` SSE is buffered (monolithic generate → UTF-8-safe 64B chunks after the fact), format-compatible, not incremental.

## Per-model config JSON (`--config`)

`--config models.json` replaces scattered CLI flags with per-model `alias` + `sampling` + `config`. Merges with CLI `--model` (both allowed); file models come first. `~/` expands via `HOME`.

```json
{
  "models": [
    {
      "path": "~/opt/models/mlx_models/Qwen3.8-27B",
      "alias": "qwen-main",
      "sampling": { "temperature": 0.7, "top_p": 0.95, "top_k": 20, "min_p": 0.05, "seed": 42, "max_tokens": 512 },
      "config": { "ctx_size": 262144, "mtp": false, "mtp_gamma": 1 }
    },
    {
      "path": "~/opt/models/mlx_models/Qwen3.8-27B-nvfp4",
      "alias": "qwen-fast",
      "sampling": { "temperature": 0.0 }
    }
  ],
  "server": { "host": "127.0.0.1", "port": 11234, "max_resident_models": 1 },
  "cache": { "prefix_cache_entries": 32, "prefix_cache_mem": "2GB", "apc_disk": "10GB", "apc_disk_dir": "/tmp/apc-disk" }
}
```

```sh
./zig-out/bin/mlx-runner --config models.json --serve --port 11234
curl http://localhost:11234/v1/models  # ["qwen-main","qwen-fast"]
curl http://localhost:11234/v1/chat/completions -d '{"model":"qwen-fast","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}'
```

- `path` required; `alias` optional (defaults to basename, must be unique). API `model` id = `alias` if present.
- `sampling` overlays that checkpoint's `generation_config.json`; `request` still wins. `max_tokens` here is the default when request omits it (server default otherwise 1024, CLI `--prompt` default 256).
- `config.ctx_size`: `0` or omitted = model max; assume RAM sufficient. `mtp`/`mtp_gamma` (γ=1 stable, γ=2 draft-ahead) per-model.
- `server`/`cache` are global defaults; CLI flags still override file.
- See `src/config.zig:loadFile` for parser, `src/server.zig:ModelSpec` for registry.

## CLI Reference

```
mlx-runner — MLX-native Metal LLM engine (Zig 0.17, Qwen3.5 GDN, no Python)

Usage: mlx-runner <command> [options]
       mlx-runner [options]

Commands:
  run <model>         Use a local checkpoint dir and chat (alias for --model)

Options:
  --model <dir>       Local checkpoint dir, repeatable for --serve
                      (default: $MLX_RUNNER_MODEL or ~/opt/models/mlx_models/Qwen3.8-27B)
  --prompt <text>     Single prompt (non-interactive)
  --chat              Chat REPL (stdin)
  --serve             Start HTTP server (OpenAI + Anthropic compat)
  --host <ip>         Bind host (default: 127.0.0.1)
  --port <n>          Bind port (default: 11234)
  --max-resident-models <n>  Serve: live engines before LRU evict (default: 1)
  --temp <f>          Temperature (default: from generation_config or 1.0)
  --top-p <f>         Top-p nucleus (default 1.0)
  --top-k <n>         Top-k (default 0 = off)
  --min-p <f>         Min-p (default 0.0)
  --seed <n>          RNG seed
  --max-tokens <n>    Max tokens to generate (default: 256)
  --ctx-size <n>      Max context length (0 = model max, 262144)
  --kv-quant <mode>   Accepted for CLI stability; only off is supported
  --stream            Stream tokens (prompt mode, v1 buffered)
  --prefix-cache-entries <n>  Hot prefix cache LRU capacity (default: 32)
  --prefix-cache-mem <size>   Hot cache byte budget (default: 2GB)
  --prefix-cache-disk <size>  SSD tier budget (default: off, e.g. 10GB)
  --apc-disk <size>           APC disk quota (default: off, e.g. 10GB)
  --apc-disk-dir <path>       APC disk location (default: ~/.mlx-runner/apc-disk/<hash>)
  --no-vision         Accepted no-op (text-only engine)
  --mtp               Enable native MTP speculative decoding (default: off)
  --no-mtp            Disable MTP
  --mtp-gamma <n>     MTP draft count (1 or 2, default: 1)
  --bench             Base-vs-MTP benchmark (needs --model); implies temp 0
  --bench-out <f>     Write bench JSON to f (table always goes to stdout)
   --tp <n>            Tensor parallel shards (stub v1, default 1)
   --pipeline <n>      Pipeline stages (stub v1, default 1)
   --config <path>     Per-model config JSON (alias + sampling + per-model ctx/mtp; see below)
   --version           Print version and exit
   --help              Show this help

Sampling priority: request > CLI > per-model config > generation_config.json > hardcoded (1.0/1.0/0/0)
Qwen3.8 defaults: temp 1.0, top_p 0.95, top_k 20.
```

Environment:

- `MLX_RUNNER_MODEL` — default `--model` if none given
- `MLX_SERVE_WIRED` — `off`/`max`/`fit` (Metal wired residency; `fit` = live + 64 MB slack, default `max`)
- `MLX_SERVE_WIRED_SLACK_MB` — slack for `fit` (default 64)
- `HOME` — used to resolve `~/` and default model/cache dirs (`~/.mlx-runner/{kv-disk,apc-disk}`)

## KV & Cache & APC

| Flag | Default | Description |
|------|---------|-------------|
| `--ctx-size` | `0` = model max (262144) | Pin max context (0 = `max_position_embeddings`; assume RAM sufficient; per-model `config.ctx_size` overrides) |
| `--prefix-cache-entries` | `32` | Hot RAM LRU (`src/cache.zig:PrefixCache` + `src/apc.zig:ApcCache`) |
| `--prefix-cache-mem` | `2GB` | Hot byte budget |
| `--prefix-cache-disk` | `off` | SSD tier for completions (`src/kv_checkpoint.zig`, `KVC\x01`, SHA1, 4096 MiB budget, 6h half-life) |
| `--apc-disk` | `off` | APC SSD quota (e.g. `10GB`, `500MB`) |
| `--apc-disk-dir` | `~/.mlx-runner/apc-disk/<hash>` | APC SSD location (absolute path) |
| `--mtp` / `--no-mtp` | `off` | Native MTP speculative decoding, γ=1 (γ=2 via `--mtp-gamma 2`) |
| `--bench` | — | Table to stdout + JSON to `--bench-out` |

**APC (Automatic Prefix Cache) — True KV reuse**

- Hot: `src/apc.zig:ApcCache` — token-id longest-prefix LRU cloning `Caches.kv`/`gdn` + `MtpState.kv` via `mlx_array_set`. Hit restores GDN `conv/ssm` + KV `keys/values/views` + `len/cap` and `MTP KV`, sets `pos=cached_len` so remaining prefill is `prompt_len - cached_len`. BPE prefix stability: byte prefix that is not token-prefix (e.g. trailing space `"... "` + `"What"` → 220 vs 3437) correctly misses; history-growth prompts always hit.
- Disk: `src/apc.zig:ApcDisk` wraps `kv_checkpoint.KVStore` but payload is safetensors bytes of KV/GDN/MTP. `store` → `KVStore.store(text, tokens, quant, model_id, ctx, payload)`, `load` → longest byte-prefix hit → deserialize → `cached_len`. Quota enforced by `KVStore.evictIfNeeded`, location via `--apc-disk-dir` or default hash. After restart, hot is cold but disk hit gives 0-compute prefill (e.g. 680 cached, 0 computed in 67 ms).
- `bypass_cache` (bench) disables both hot and disk.

Example (nvfp4, 680 tok, striped prompt):

```sh
# 1GB quota, explicit location
./zig-out/bin/mlx-runner --serve --model ~/opt/models/mlx_models/Qwen3.8-27B-nvfp4 \
  --apc-disk 1GB --apc-disk-dir /tmp/apc-disk

# First: system alone (20× paragraph) → miss 679 tok in 919 ms, stores 342M file
# Second: same + " What is 2+2?" → hit 679 cached +7 computed in 207 ms (11×)
# After restart: same second prompt → disk hit 679 cached, 0 computed in 67 ms
```

## Quantization

- Per-checkpoint `WeightMap.quant` inferred from shapes: `U8 scales` + `packed/scales` ratio `r` → `r=8` mxfp8 (32,8), `r=2` nvfp4 (16,4); verified against `mlx` `_defaults_for_mode`.
- Forward: `namedLinearQ`/`layerLinearQ`/`mlpForwardQ` + `mtpLinearQ`; dense fallback iff scales absent and weight not `U32`.
- Synthetic test (MLX quantize → `qmm` ≈ CPU matmul over dequantize, tol 0.05) + live coherent 40-token nvfp4 output.

## Layout

```
src/main.zig          — CLI (std.process.Init + std.Io), repeatable --model + --config
src/config.zig        — per-model config JSON (alias/sampling/ctx/mtp, ~/ expand)
src/engine.zig        — Engine: chunked prefill, base/MTP decode, hot/disk/APC
src/model.zig         — qwen3_5 forward (GDN + full attn + MLP; fused Metal kernel)
src/gdn_packed.metal  — packed GDN recurrence (vendored from mlx_lm, T as uint32[1])
src/mtp.zig           — MTP head (fc + 1 decoder + norm + shared lm_head)
src/apc.zig           — APC hot + disk (KV/GDN/MTP clone/restore, safetensors)
src/cache.zig         — RAM LRU (PrefixCache)
src/kv_checkpoint.zig — SSD fixed checkpoint (KVC, SHA1, budget, eviction)
src/tokenizer.zig     — byte-level BPE (no regex engine)
src/weights.zig       — sharded safetensors loader (bf16/mxfp8/nvfp4)
src/sampling.zig      — SamplingParams + generation_config.json
src/sample.zig        — temp + top-k/top-p/min-p + categorical
src/bench.zig         — --bench implementation
src/server.zig        — HTTP glue + ModelRegistry (lazy/evict)
src/http_api.zig      — pure protocol builders + wire-format tests (hermetic)
src/mlx.zig           — mlx-c FFI
src/tests.zig         — hermetic suite; src/mlx_test.zig — linked (Metal) suite
```

## Testing

```sh
.zig-toolchain/zig build test --summary all           # hermetic, no Metal
.zig-toolchain/zig build test-mlx --summary all        # linked, needs lib/mlx + Metal
make test && make test-mlx
./zig-out/bin/mlx-runner --metal-check                # Metal + mlx-c smoke

# Live (needs local checkpoint)
MODEL=~/opt/models/mlx_models/Qwen3.8-27B
./zig-out/bin/mlx-runner --model $MODEL --prompt "The capital of France is" --temp 0 --max-tokens 10
./zig-out/bin/mlx-runner --model $MODEL --prompt "Hi" --mtp --max-tokens 32 | head
./zig-out/bin/mlx-runner --model $MODEL --bench --bench-out bench.json
```

Hermetic: `sampling`, `cache`, `tokenizer`, `weights`, `http_api`, `kv_checkpoint`, `config`. Linked: `model`, `sample`, `mtp`, `engine`, `bench`, `apc` (Metal, `libmlx`).

## Versioning & Release

- Version source: `build.zig.zon:.version` (e.g. `0.1.0`) and `src/main.zig:VERSION` (e.g. `0.1.0-zig`); keep in sync.
- Tag format: `v<version>` (e.g. `v0.1.0`). CI creates tag if missing.
- **Tag workflow** (`.github/workflows/tag.yml`): on push to `main`, reads version from `build.zig.zon`, checks if tag `v<version>` exists, creates and pushes it if new. This triggers the release workflow.
- **Release workflow** (`.github/workflows/release.yml`): on push tag `v*` (or manual dispatch), builds `mlx + mlx-c` + `mlx-runner` (ReleaseFast), smoke-tests, packages `mlx-runner-macos-arm64.{zip,tar.gz}` + `checksums.txt` (re-wired rpaths), and creates a GitHub Release with those assets (via `softprops/action-gh-release`). Dispatch also uploads artifacts.

```sh
# Bump version
# 1) edit build.zig.zon:.version and src/main.zig:VERSION
# 2) commit & push to main → tag workflow creates vX.Y.Z if missing → release workflow builds & publishes
# Manual:
git tag v0.2.0 && git push origin v0.2.0   # triggers release directly
```

## References

- Model card: https://huggingface.co/Qwen/Qwen3.8-27B (native VLM: text, image, video; thinking mode default; 262144 ctx)
- `mlx-lm` main: `models/qwen3_5.py` (GDN/decoder/sanitize), `models/qwen3_next.py` (attention/MLP/norm), `models/gated_delta.py` (recurrence), `models/base.py` (masks), `models/cache.py` (KVCache)
- `AirRunner/mlx-lm` `feat/mtp-native`: `models/qwen3_5.py` (MTP head), `generate.py` (`mtp_generate_step`, rollback)
- NAX dispatch: https://github.com/ml-explore/mlx/issues/3182 (`is_nax_available`: macOS ≥ 26.2 + arch gen ≥ 17; automatic in MLX)
- `~/projects/llm-proj/mlx-infer`: Zig sampler idioms (`generate.zig`)

## License

MIT — see `LICENSE`.
