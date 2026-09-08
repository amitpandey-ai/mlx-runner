#!/usr/bin/env bash
# Build mlx + mlx-c for mlx-runner (no Python anywhere).
# - If lib/mlx-src + lib/mlxc-src submodules exist, do full NAX build (like mlx-infer/scripts/build-mlx.sh, requires Xcode 26.2 + MetalToolchain)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAGE="$REPO_ROOT/lib/mlx"
if [ -f "$REPO_ROOT/lib/mlx-src/CMakeLists.txt" ] && [ -f "$REPO_ROOT/lib/mlxc-src/CMakeLists.txt" ]; then
  echo "[build-mlx] submodules present — delegating to mlx-infer style NAX build"
  # Reuse mlx-infer's script if available, else inline
  if [ -f "$REPO_ROOT/../mlx-infer/scripts/build-mlx.sh" ]; then
    # Build mlx-infer's stage first, then copy
    bash "$REPO_ROOT/../mlx-infer/scripts/build-mlx.sh"
    rm -rf "$STAGE"
    cp -R "$REPO_ROOT/../mlx-infer/lib/mlx" "$STAGE"
    echo "[build-mlx] copied NAX stage from mlx-infer/lib/mlx"
    exit 0
  fi
  # fallback: direct build (same as mlx-infer)
  DEPLOYMENT_TARGET="${MLX_DEPLOYMENT_TARGET:-26.2}"
  MLX_SRC="$REPO_ROOT/lib/mlx-src"
  MLXC_SRC="$REPO_ROOT/lib/mlxc-src"
  BUILD_ROOT="$REPO_ROOT/lib/.mlx-build"
  STAMP="$STAGE/.version"
  xcrun -sdk macosx metal --version >/dev/null 2>&1 || { echo "Metal toolchain missing — xcodebuild -downloadComponent MetalToolchain"; exit 1; }
  SDK_VERSION="$(xcrun --sdk macosx --show-sdk-version)"
  echo "[build-mlx] SDK $SDK_VERSION target $DEPLOYMENT_TARGET"
  NCPU="$(sysctl -n hw.ncpu)"
  cmake -S "$MLX_SRC" -B "$BUILD_ROOT/mlx" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" -DBUILD_SHARED_LIBS=ON -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="$STAGE"
  cmake --build "$BUILD_ROOT/mlx" -j "$NCPU"
  cmake --install "$BUILD_ROOT/mlx" >/dev/null
  cmake -S "$MLXC_SRC" -B "$BUILD_ROOT/mlxc" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DMLX_C_USE_SYSTEM_MLX=ON -DCMAKE_PREFIX_PATH="$STAGE" -DCMAKE_INSTALL_PREFIX="$STAGE"
  cmake --build "$BUILD_ROOT/mlxc" -j "$NCPU"
  cmake --install "$BUILD_ROOT/mlxc" >/dev/null
  exit 0
fi

echo "[build-mlx] no submodules — staging native mlx (no Python, Metal still works)"
# NOTE: since mlx 0.32 the PyPI `mlx` wheel is a thin Python shim (.so only);
# the native bits (libmlx.dylib, mlx.metallib, headers) ship in `mlx-metal`.
# NAX (M5 matrix accelerator, ~3.7x bf16 GEMM measured): the PyPI wheels are
# built without NAX kernels, so prefer a local NAX-enabled mlx when one
# exists at the pinned version; otherwise fall back to the wheel (works,
# just not on NAX). Building NAX from source needs Xcode 26.2+ (Metal 4),
# which this machine lacks — hence the local-binary preference.
MLX_VERSION="${MLX_VERSION:-0.32.2}"
MTPLX_MLX="/opt/homebrew/var/mtplx/venv-2.10.2/lib/python3.13/site-packages/mlx"
USE_LOCAL_NAX=""
if [ -f "$MTPLX_MLX/lib/libmlx.dylib" ] && [ -f "$MTPLX_MLX/lib/mlx.metallib" ] && [ -d "$MTPLX_MLX/include" ]; then
  if grep -q "^Version: ${MLX_VERSION}$" "$MTPLX_MLX/../mlx-${MLX_VERSION}.dist-info/METADATA" 2>/dev/null; then
    if strings -a "$MTPLX_MLX/lib/mlx.metallib" 2>/dev/null | grep -q "steel_gemm_fused_nax"; then
      USE_LOCAL_NAX="$MTPLX_MLX"
    fi
  fi
fi
if [ -n "$USE_LOCAL_NAX" ]; then
  echo "[build-mlx] staging NAX-enabled mlx==$MLX_VERSION from $USE_LOCAL_NAX"
  mkdir -p "$STAGE/lib" "$STAGE/include"
  cp "$USE_LOCAL_NAX/lib/libmlx.dylib" "$USE_LOCAL_NAX/lib/mlx.metallib" "$STAGE/lib/"
  cp "$USE_LOCAL_NAX/lib/libjaccl.dylib" "$STAGE/lib/" 2>/dev/null || true
  cp -R "$USE_LOCAL_NAX/include/"* "$STAGE/include/"
  if [ -d "$USE_LOCAL_NAX/share/cmake" ]; then
    mkdir -p "$STAGE/share"
    cp -R "$USE_LOCAL_NAX/share/cmake" "$STAGE/share/" 2>/dev/null || true
  fi
  if [ -d "$USE_LOCAL_NAX/lib/cmake" ]; then
    cp -R "$USE_LOCAL_NAX/lib/cmake" "$STAGE/lib/" 2>/dev/null || true
  fi
  echo "mlx==$MLX_VERSION NAX from $USE_LOCAL_NAX" > "$STAGE/.version.src"
else
echo "[build-mlx] no local NAX mlx==$MLX_VERSION — wheel fallback (non-NAX)"
WHEEL_TMP="$(mktemp -d)"
trap 'rm -rf "$WHEEL_TMP"' EXIT
# Try pip download first (more reliable than scraping Simple HTML)
if command -v pip3 >/dev/null 2>&1; then
  echo "[build-mlx] trying pip3 download mlx-metal==$MLX_VERSION"
  pip3 download --only-binary=:all: --no-deps -d "$WHEEL_TMP" "mlx-metal==$MLX_VERSION" 2>&1 | tail -n 20 || true
  WHEEL_FILE="$(ls "$WHEEL_TMP"/mlx_metal-*.whl 2>/dev/null | head -1)"
  if [ -n "$WHEEL_FILE" ] && [ ! -f "$WHEEL_TMP/mlx.whl" ]; then
    echo "[build-mlx] pip3 downloaded $WHEEL_FILE"
    cp "$WHEEL_FILE" "$WHEEL_TMP/mlx.whl"
  fi
fi
# Fallback to scraping Simple index if pip download failed
if [ ! -f "$WHEEL_TMP/mlx.whl" ]; then
  echo "[build-mlx] pip download failed, scraping https://pypi.org/simple/mlx-metal/"
  SIMPLE_HTML="$(curl -fSL --retry 3 "https://pypi.org/simple/mlx-metal/" || true)"
  WHEEL_FILE="$(printf '%s' "$SIMPLE_HTML" | grep -o "mlx_metal-${MLX_VERSION}[^\"]*macosx_14_0_arm64\.whl" | head -1)"
  if [ -z "$WHEEL_FILE" ]; then
    WHEEL_FILE="$(printf '%s' "$SIMPLE_HTML" | grep -o "mlx_metal-${MLX_VERSION}[^\"]*macosx_[^\"]*arm64\.whl" | head -1)"
  fi
  if [ -z "$WHEEL_FILE" ]; then echo "[build-mlx] ERROR: no macOS arm64 wheel for mlx-metal==${MLX_VERSION} on PyPI" >&2; ls -la "$WHEEL_TMP" >&2; echo "$SIMPLE_HTML" | head -n 20 >&2; exit 1; fi
  case "$WHEEL_FILE" in *"$MLX_VERSION"*) ;; *) echo "[build-mlx] ERROR: resolved wheel '$WHEEL_FILE' lacks version $MLX_VERSION" >&2; exit 1;; esac
  echo "[build-mlx] resolved wheel $WHEEL_FILE via scrape"
  WHEEL_URL="$(printf '%s' "$SIMPLE_HTML" | grep -o "https://files.pythonhosted.org[^\"]*${WHEEL_FILE}[^\"]*" | head -1)"
  if [ -z "$WHEEL_URL" ]; then echo "[build-mlx] ERROR: no URL for $WHEEL_FILE" >&2; exit 1; fi
  curl -fSL --retry 3 -o "$WHEEL_TMP/mlx.whl" "$WHEEL_URL"
fi
if [ ! -f "$WHEEL_TMP/mlx.whl" ]; then echo "[build-mlx] ERROR: no wheel downloaded" >&2; ls -lh "$WHEEL_TMP" >&2; exit 1; fi
echo "[build-mlx] wheel ready $(ls -lh "$WHEEL_TMP/mlx.whl" | awk '{print $9, $5}')"
rm -rf "$WHEEL_TMP/unpack"
mkdir -p "$WHEEL_TMP/unpack"
unzip -q "$WHEEL_TMP/mlx.whl" -d "$WHEEL_TMP/unpack"
LIBMLX="$(find "$WHEEL_TMP/unpack" -name libmlx.dylib | head -1)"
METALLIB="$(find "$WHEEL_TMP/unpack" -name mlx.metallib | head -1)"
MLX_PKG="$(dirname "$(dirname "$LIBMLX")")"
if [ -z "$LIBMLX" ] || [ ! -f "$LIBMLX" ]; then echo "[build-mlx] ERROR: libmlx.dylib not in wheel" >&2; exit 1; fi
if [ -z "$METALLIB" ] || [ ! -f "$METALLIB" ]; then echo "[build-mlx] ERROR: mlx.metallib not in wheel" >&2; exit 1; fi
mkdir -p "$STAGE/lib" "$STAGE/include"
cp "$LIBMLX" "$STAGE/lib/"
cp "$METALLIB" "$STAGE/lib/"
JACCL="$(find "$WHEEL_TMP/unpack" -name "libjaccl.dylib" | head -1)"
if [ -n "$JACCL" ]; then cp "$JACCL" "$STAGE/lib/" 2>/dev/null || true; fi
if [ -d "$MLX_PKG/include" ]; then
  cp -R "$MLX_PKG/include/"* "$STAGE/include/" 2>/dev/null || true
elif [ -d "$WHEEL_TMP/unpack/mlx/include" ]; then
  cp -R "$WHEEL_TMP/unpack/mlx/include/"* "$STAGE/include/" 2>/dev/null || true
fi
# CMake package configs so find_package(MLX) resolves the staged 0.32.2
# instead of a system/brew mlx (header skew breaks the mlx-c build).
if [ -d "$MLX_PKG/share/cmake" ]; then
  mkdir -p "$STAGE/share"
  cp -R "$MLX_PKG/share/cmake" "$STAGE/share/" 2>/dev/null || true
fi
if [ -d "$MLX_PKG/lib/cmake" ]; then
  cp -R "$MLX_PKG/lib/cmake" "$STAGE/lib/" 2>/dev/null || true
fi
rm -rf "$WHEEL_TMP"
trap - EXIT
fi
# Need mlxc headers
if [ -d "$REPO_ROOT/../mlx-infer/lib/mlxc-src/mlx/c" ]; then
  mkdir -p "$STAGE/include/mlx/c"
  cp "$REPO_ROOT/../mlx-infer/lib/mlxc-src/mlx/c/"*.h "$STAGE/include/mlx/c/"
elif [ ! -f "$STAGE/include/mlx/c/mlx.h" ]; then
  echo "[build-mlx] cloning mlx-c for headers"
  tmp="$(mktemp -d)"
  git clone --depth 1 https://github.com/ml-explore/mlx-c "$tmp/mlx-c"
  mkdir -p "$STAGE/include/mlx/c"
  cp "$tmp/mlx-c/mlx/c/"*.h "$STAGE/include/mlx/c/"
  rm -rf "$tmp"
fi
# Build mlxc against staged mlx
MLXC_SRC_TMP="$(mktemp -d)"
if [ -d "$REPO_ROOT/../mlx-infer/lib/mlxc-src" ]; then
  cp -R "$REPO_ROOT/../mlx-infer/lib/mlxc-src" "$MLXC_SRC_TMP/mlx-c"
else
  git clone --depth 1 https://github.com/ml-explore/mlx-c "$MLXC_SRC_TMP/mlx-c"
fi
BUILD_DIR="$REPO_ROOT/lib/.mlx-build/mlxc-pip"
rm -rf "$BUILD_DIR"
cmake -S "$MLXC_SRC_TMP/mlx-c" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DMLX_C_USE_SYSTEM_MLX=ON -DCMAKE_PREFIX_PATH="$STAGE" -DCMAKE_INSTALL_PREFIX="$STAGE" >/dev/null
cmake --build "$BUILD_DIR" -j "$(sysctl -n hw.ncpu)" >/dev/null
cmake --install "$BUILD_DIR" >/dev/null 2>&1 || cp "$BUILD_DIR"/libmlxc.dylib "$STAGE/lib/" 2>/dev/null || cp "$BUILD_DIR"/*.dylib "$STAGE/lib/" 2>/dev/null || true
rm -rf "$MLXC_SRC_TMP" "$BUILD_DIR"
if [ -n "${WHEEL_FILE:-}" ]; then
  echo "mlx==$MLX_VERSION wheel $WHEEL_FILE mlxc wheel-fallback" > "$STAGE/.version"
elif [ -f "$STAGE/.version.src" ]; then
  mv "$STAGE/.version.src" "$STAGE/.version"
else
  echo "mlxc fallback" > "$STAGE/.version"
fi
echo "[build-mlx] staged $STAGE/lib: $(ls -lh "$STAGE/lib" | awk '{print $9, $5}')"
NAXN="$(strings "$STAGE/lib/mlx.metallib" 2>/dev/null | grep -c "steel_gemm_fused_nax" || true)"
echo "[build-mlx] NAX kernels: ${NAXN:-0} (0 = non-NAX build, still Metal)"
