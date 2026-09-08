#!/usr/bin/env bash
# mlx-runner installer — latest GitHub Release for macOS (Apple Silicon)
# Usage: curl -fsSL https://raw.githubusercontent.com/amitpandey-ai/mlx-runner/main/install.sh | bash
#        curl -fsSL https://raw.githubusercontent.com/amitpandey-ai/mlx-runner/main/install.sh | bash -s -- --prefix /usr/local --version v0.1.0
set -euo pipefail

REPO="amitpandey-ai/mlx-runner"
ASSET_TGZ="mlx-runner-macos-arm64.tar.gz"
ASSET_ZIP="mlx-runner-macos-arm64.zip"
CHECKSUMS="checksums.txt"

PREFIX_DEFAULT="${HOME:-/tmp}/.local"
PREFIX="${PREFIX:-$PREFIX_DEFAULT}"
BINDIR="${BINDIR:-}"
LIBDIR="${LIBDIR:-}"
TAG=""
UNINSTALL=0
FORCE=0
DRY_RUN=0
VERBOSE=0

info()  { printf "[install] %s\n" "$*"; }
warn()  { printf "[install] WARN: %s\n" "$*" >&2; }
err()   { printf "[install] ERROR: %s\n" "$*" >&2; }
verbose() { [ "$VERBOSE" -eq 1 ] && printf "[install] %s\n" "$*" || true; }

usage() {
  cat <<EOF
mlx-runner installer — macOS arm64 (latest release)

Usage: install.sh [options]
       curl -fsSL https://raw.githubusercontent.com/$REPO/main/install.sh | bash
       curl -fsSL https://raw.githubusercontent.com/$REPO/main/install.sh | bash -s -- [options]

Options:
  --prefix DIR        Install prefix (default: \$HOME/.local or PREFIX env)
                      -> binary: \$PREFIX/bin/mlx-runner
                      -> libs:   \$PREFIX/lib/mlx-runner
  --bindir DIR        Override binary dir (default: \$PREFIX/bin)
  --libdir DIR        Override library dir (default: \$PREFIX/lib/mlx-runner)
  --version TAG       Install specific tag (e.g. v0.1.0 or 0.1.0, default: latest)
  --tag TAG           Alias for --version
  --force             Overwrite existing install without prompt
  --dry-run           Show what would be done, don't write
  --verbose           Verbose output
  --uninstall         Remove installed binary + libs (uses --prefix/--bindir/--libdir)
  --help, -h          Show this help

Env:
  PREFIX, BINDIR, LIBDIR, GITHUB_TOKEN (for higher API rate limit)
  VERSION / TAG        Alternative to --version

Examples:
  curl -fsSL https://raw.githubusercontent.com/$REPO/main/install.sh | bash
  curl -fsSL https://raw.githubusercontent.com/$REPO/main/install.sh | bash -s -- --prefix /usr/local
  ./install.sh --version v0.1.0 --verbose
  ./install.sh --uninstall
EOF
}

# Parse args
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="${2:-}"; shift 2 ;;
    --prefix=*) PREFIX="${1#*=}"; shift ;;
    --bindir) BINDIR="${2:-}"; shift 2 ;;
    --bindir=*) BINDIR="${1#*=}"; shift ;;
    --libdir) LIBDIR="${2:-}"; shift 2 ;;
    --libdir=*) LIBDIR="${1#*=}"; shift ;;
    --version) TAG="${2:-}"; shift 2 ;;
    --version=*) TAG="${1#*=}"; shift ;;
    --tag) TAG="${2:-}"; shift 2 ;;
    --tag=*) TAG="${1#*=}"; shift ;;
    --force|-f) FORCE=1; shift ;;
    --dry-run|--dryrun|-n) DRY_RUN=1; shift ;;
    --verbose|-v) VERBOSE=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    -*) err "unknown flag $1"; usage; exit 1 ;;
    *) err "unexpected arg $1"; usage; exit 1 ;;
  esac
done

# Env fallbacks for tag
if [ -z "$TAG" ]; then
  if [ -n "${VERSION:-}" ]; then TAG="$VERSION"
  elif [ -n "${TAG_ENV:-}" ]; then TAG="$TAG_ENV"
  fi
fi

# Normalize dirs: expand ~ and strip trailing slash
expand_path() {
  local p="$1"
  case "$p" in
    "~"/*) p="${HOME}${p#\~}" ;;
    "~") p="$HOME" ;;
  esac
  # remove trailing slash
  p="${p%/}"
  printf "%s" "$p"
}
PREFIX="$(expand_path "$PREFIX")"
if [ -z "$BINDIR" ]; then BINDIR="$PREFIX/bin"; else BINDIR="$(expand_path "$BINDIR")"; fi
if [ -z "$LIBDIR" ]; then LIBDIR="$PREFIX/lib/mlx-runner"; else LIBDIR="$(expand_path "$LIBDIR")"; fi

# Uninstall path
if [ "$UNINSTALL" -eq 1 ]; then
  info "uninstalling mlx-runner"
  info "  bin: $BINDIR/mlx-runner"
  info "  lib: $LIBDIR"
  if [ "$DRY_RUN" -eq 1 ]; then
    info "(dry-run) would remove above"
    exit 0
  fi
  rm -f "$BINDIR/mlx-runner"
  rm -rf "$LIBDIR"
  info "uninstalled (if present)"
  exit 0
fi

# Preflight: OS / arch
OS="$(uname -s)"
ARCH="$(uname -m)"
if [ "$OS" != "Darwin" ]; then
  err "unsupported OS $OS — mlx-runner requires macOS (Darwin) on Apple Silicon"
  err "  To build from source on Linux, use: bash scripts/build-mlx.sh && .zig-toolchain/zig build -Doptimize=ReleaseFast"
  exit 1
fi
case "$ARCH" in
  arm64|aarch64) ;;
  *)
    err "unsupported arch $ARCH — mlx-runner releases are arm64-only"
    err "  Intel Macs: build from source (non-NAX, slower) via: make build"
    exit 1
    ;;
esac

# Warn on old macOS (<26.2) — NAX needs 26.2+ but binary still runs without NAX
if command -v sw_vers >/dev/null 2>&1; then
  ver="$(sw_vers -productVersion 2>/dev/null || echo "0.0")"
  major="$(printf "%s" "$ver" | cut -d. -f1)"
  minor="$(printf "%s" "$ver" | cut -d. -f2)"
  # need major 26+ or (26 + minor >=2). For simplicity warn if major <15 (Sequoia) or major==26 with minor<2 pre-release.
  # Actually macOS 15 is Sequoia, 26 is Tahoe. Warn if <14.
  verbose "macOS $ver detected"
  # Require at least macOS 14; NAX optimal on 26.2+
  if [ "${major:-0}" -lt 14 ] 2>/dev/null; then
    warn "macOS $ver is old; mlx-runner needs 26.2+ for NAX. Install will proceed but Metal may fail."
  elif [ "$major" -eq 26 ] && [ "${minor:-0}" -lt 2 ] 2>/dev/null; then
    warn "macOS $ver < 26.2 — NAX unavailable, but install will proceed"
  fi
fi

# Tools check
need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    err "missing required tool: $1"
    case "$1" in
      curl) err "  install curl (macOS ships it) or pass --version with manual download" ;;
      tar) err "  tar is required" ;;
      unzip|ditto) err "  unzip/ditto required for zip fallback" ;;
    esac
    exit 1
  fi
}
need_cmd curl
need_cmd tar
# ditto is preferred on macOS for zip, fallback unzip
if ! command -v ditto >/dev/null 2>&1 && ! command -v unzip >/dev/null 2>&1; then
  err "need ditto or unzip to handle zip fallback"
  exit 1
fi
if ! command -v install_name_tool >/dev/null 2>&1; then
  warn "install_name_tool not found — rpath fix may be skipped (install may still work if release layout matches)"
fi
if ! command -v otool >/dev/null 2>&1; then
  warn "otool not found — rpath detection skipped"
fi

# Resolve TAG if not given
normalize_tag() {
  local t="$1"
  # strip whitespace
  t="$(printf "%s" "$t" | tr -d ' \n\r\t')"
  case "$t" in
    v*) printf "%s" "$t" ;;
    "") printf "" ;;
    *) printf "v%s" "$t" ;;
  esac
}
if [ -z "$TAG" ]; then
  info "resolving latest release tag for $REPO..."
  TAG_RAW=""
  # Try GitHub API (with optional token)
  AUTH_HDR=""
  if [ -n "${GITHUB_TOKEN:-}" ]; then AUTH_HDR="Authorization: token $GITHUB_TOKEN"; fi
  # curl API, parse tag_name
  if [ -n "$AUTH_HDR" ]; then
    TAG_RAW="$(curl -fsSL -H "$AUTH_HDR" "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null | grep -o '"tag_name"[[:space:]]*:[[:space:]]*"[^"]*"' | sed -E 's/.*"([^"]+)".*/\1/' || true)"
  else
    TAG_RAW="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null | grep -o '"tag_name"[[:space:]]*:[[:space:]]*"[^"]*"' | sed -E 's/.*"([^"]+)".*/\1/' || true)"
  fi
  if [ -n "$TAG_RAW" ]; then
    TAG="$(normalize_tag "$TAG_RAW")"
    verbose "API resolved tag $TAG"
  fi
  # Fallback: follow redirect of /releases/latest
  if [ -z "$TAG" ]; then
    verbose "API failed or rate-limited, trying redirect"
    EFFECTIVE="$(curl -fsSL -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest" 2>/dev/null || true)"
    # effective like https://github.com/amitpandey-ai/mlx-runner/releases/tag/v0.1.0
    TAG_FALLBACK="$(printf "%s" "$EFFECTIVE" | sed -n 's/.*\/tag\/\(v[^\/]*\).*/\1/p')"
    if [ -n "$TAG_FALLBACK" ]; then
      TAG="$TAG_FALLBACK"
      verbose "redirect resolved tag $TAG from $EFFECTIVE"
    fi
  fi
  if [ -z "$TAG" ]; then
    err "failed to resolve latest tag (API rate limit or no releases)"
    err "  Try: ./install.sh --version v0.1.0"
    err "  Or set GITHUB_TOKEN env for higher limit"
    exit 1
  fi
else
  TAG="$(normalize_tag "$TAG")"
fi

info "installing mlx-runner $TAG"
info "  prefix: $PREFIX"
info "  bin:    $BINDIR/mlx-runner"
info "  lib:    $LIBDIR"

if [ "$DRY_RUN" -eq 1 ]; then
  info "(dry-run) would download $TAG"
  info "  https://github.com/$REPO/releases/download/$TAG/$ASSET_TGZ"
  info "  or https://github.com/$REPO/releases/download/$TAG/$ASSET_ZIP"
  info "  and install to $BINDIR / $LIBDIR (skipping network in dry-run)"
  exit 0
fi

# Check writable
check_writable() {
  local dir="$1"
  local base="$dir"
  # walk up until existing
  while [ ! -e "$base" ] && [ "$base" != "/" ] && [ "$base" != "." ]; do
    base="$(dirname "$base")"
  done
  if [ ! -w "$base" ] 2>/dev/null; then
    warn "$dir not writable (parent $base). You may need: sudo mkdir -p \"$dir\"; sudo chown \$(id -un) \"$dir\""
    if [ "$PREFIX" = "/usr/local" ] || [ "$PREFIX" = "/opt/homebrew" ]; then
      warn "  Try: sudo ./install.sh --prefix $PREFIX  (or chown prefix first)"
    fi
  fi
}
check_writable "$BINDIR"
check_writable "$LIBDIR"

# Check existing install
if [ -x "$BINDIR/mlx-runner" ] && [ "$FORCE" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
  existing_ver="$("$BINDIR/mlx-runner" --version 2>/dev/null | head -1 || echo "unknown")"
  info "existing install: $existing_ver at $BINDIR/mlx-runner (use --force to overwrite)"
fi

# Download
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
verbose "tmp $TMP"

BASE_URL="https://github.com/$REPO/releases/download/$TAG"
URL_TGZ="$BASE_URL/$ASSET_TGZ"
URL_ZIP="$BASE_URL/$ASSET_ZIP"
URL_SUMS="$BASE_URL/$CHECKSUMS"

ARCHIVE=""
ARCHIVE_TYPE=""

download_file() {
  local url="$1" dest="$2"
  verbose "downloading $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --progress-bar -o "$dest" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$dest" "$url"
  else
    err "no curl or wget"
    return 1
  fi
}

# Try tgz first, then zip
set +e
download_file "$URL_TGZ" "$TMP/$ASSET_TGZ"
RC=$?
set -e
if [ $RC -eq 0 ] && [ -s "$TMP/$ASSET_TGZ" ]; then
  ARCHIVE="$TMP/$ASSET_TGZ"
  ARCHIVE_TYPE="tgz"
  info "downloaded $ASSET_TGZ ($(du -h "$ARCHIVE" | awk '{print $1}'))"
else
  warn "tar.gz not found at $URL_TGZ (trying zip)"
  rm -f "$TMP/$ASSET_TGZ"
  set +e
  download_file "$URL_ZIP" "$TMP/$ASSET_ZIP"
  RC=$?
  set -e
  if [ $RC -eq 0 ] && [ -s "$TMP/$ASSET_ZIP" ]; then
    ARCHIVE="$TMP/$ASSET_ZIP"
    ARCHIVE_TYPE="zip"
    info "downloaded $ASSET_ZIP ($(du -h "$ARCHIVE" | awk '{print $1}'))"
  else
    err "failed to download $ASSET_TGZ or $ASSET_ZIP for $TAG"
    err "  URLs tried:"
    err "    $URL_TGZ"
    err "    $URL_ZIP"
    err "  Check tag exists: https://github.com/$REPO/releases/tag/$TAG"
    exit 1
  fi
fi

# Checksums (optional)
if [ "$DRY_RUN" -eq 0 ]; then
  verbose "fetching checksums $URL_SUMS"
  set +e
  download_file "$URL_SUMS" "$TMP/$CHECKSUMS" 2>/dev/null
  RC=$?
  set -e
  if [ $RC -eq 0 ] && [ -s "$TMP/$CHECKSUMS" ]; then
    info "verifying checksum"
    # Prefer shasum, fallback sha256sum, openssl
    SHA_CMD=""
    if command -v shasum >/dev/null 2>&1; then SHA_CMD="shasum -a 256"
    elif command -v sha256sum >/dev/null 2>&1; then SHA_CMD="sha256sum"
    elif command -v openssl >/dev/null 2>&1; then SHA_CMD="openssl dgst -sha256"
    fi
    if [ -n "$SHA_CMD" ]; then
      # checksums.txt has lines like "<sha>  <file>"
      # Try native -c if available (shasum/ sha256sum support -c)
      verified=0
      # Extract expected for our archive
      expected="$(grep -E " $(basename "$ARCHIVE")\$| $(basename "$ARCHIVE")[[:space:]]" "$TMP/$CHECKSUMS" 2>/dev/null | awk '{print $1}' | head -1 || true)"
      if [ -n "$expected" ]; then
        verbose "expected $expected"
        if [ "$SHA_CMD" = "shasum -a 256" ] || [ "$SHA_CMD" = "sha256sum" ]; then
          actual="$($SHA_CMD "$ARCHIVE" 2>/dev/null | awk '{print $1}')"
        else
          actual="$(openssl dgst -sha256 "$ARCHIVE" 2>/dev/null | awk '{print $NF}')"
        fi
        verbose "actual   $actual"
        if [ "$actual" = "$expected" ]; then
          info "checksum OK ($actual)"
          verified=1
        else
          err "checksum mismatch for $(basename "$ARCHIVE")"
          err "  expected $expected"
          err "  actual   $actual"
          if [ "$FORCE" -eq 1 ]; then
            warn "--force: ignoring checksum failure"
          else
            exit 1
          fi
        fi
      else
        warn "checksum entry for $(basename "$ARCHIVE") not in $CHECKSUMS — skipping verify"
        cat "$TMP/$CHECKSUMS" | head -n 20 >&2 || true
      fi
    else
      warn "no shasum/sha256sum/openssl — skipping verify"
    fi
  else
    warn "no checksums.txt at $URL_SUMS — skipping verify"
  fi
fi

if [ "$DRY_RUN" -eq 1 ]; then
  info "(dry-run) would extract $ARCHIVE and install to $BINDIR / $LIBDIR"
  exit 0
fi

info "extracting $ARCHIVE"
mkdir -p "$TMP/extract"
if [ "$ARCHIVE_TYPE" = "tgz" ]; then
  tar -xzf "$ARCHIVE" -C "$TMP/extract"
else
  if command -v ditto >/dev/null 2>&1; then
    # ditto handles zip with parent preservation; unzip to extract dir
    # Use ditto -xk for zip
    ditto -xk "$ARCHIVE" "$TMP/extract" 2>/dev/null || unzip -q "$ARCHIVE" -d "$TMP/extract"
  else
    unzip -q "$ARCHIVE" -d "$TMP/extract"
  fi
fi

# Find staged binary: prefer .../mlx-runner-macos-arm64/mlx-runner else any mlx-runner
STAGED_BIN=""
STAGED_LIB_DIR=""
# Common: $TMP/extract/mlx-runner-macos-arm64/mlx-runner
if [ -x "$TMP/extract/mlx-runner-macos-arm64/mlx-runner" ]; then
  STAGED_BIN="$TMP/extract/mlx-runner-macos-arm64/mlx-runner"
  if [ -d "$TMP/extract/mlx-runner-macos-arm64/lib" ]; then
    STAGED_LIB_DIR="$TMP/extract/mlx-runner-macos-arm64/lib"
  fi
fi
# Also check flat
if [ -z "$STAGED_BIN" ]; then
  STAGED_BIN="$(find "$TMP/extract" -type f -name "mlx-runner" -perm -111 2>/dev/null | head -1 || true)"
  if [ -z "$STAGED_BIN" ]; then
    STAGED_BIN="$(find "$TMP/extract" -type f -name "mlx-runner" 2>/dev/null | head -1 || true)"
  fi
  if [ -n "$STAGED_BIN" ]; then
    cand_lib="$(dirname "$STAGED_BIN")/lib"
    if [ -d "$cand_lib" ]; then STAGED_LIB_DIR="$cand_lib"
    else
      # search for libmlx.dylib location
      found_lib="$(find "$TMP/extract" -name "libmlx.dylib" 2>/dev/null | head -1 || true)"
      if [ -n "$found_lib" ]; then STAGED_LIB_DIR="$(dirname "$found_lib")"; fi
    fi
  fi
fi

if [ -z "$STAGED_BIN" ] || [ ! -f "$STAGED_BIN" ]; then
  err "extracted archive missing mlx-runner binary"
  find "$TMP/extract" -type f | head -n 20 >&2 || true
  exit 1
fi
if [ -z "$STAGED_LIB_DIR" ] || [ ! -d "$STAGED_LIB_DIR" ]; then
  # fallback: find any lib dir
  STAGED_LIB_DIR="$(find "$TMP/extract" -type d -name "lib" 2>/dev/null | head -1 || true)"
fi
if [ -z "$STAGED_LIB_DIR" ] || [ ! -f "$STAGED_LIB_DIR/libmlx.dylib" ]; then
  warn "lib dir not found or missing libmlx.dylib — continuing with binary only"
  STAGED_LIB_DIR=""
fi

verbose "staged bin $STAGED_BIN"
verbose "staged lib $STAGED_LIB_DIR"

info "installing to $BINDIR and $LIBDIR"
mkdir -p "$BINDIR" "$LIBDIR"
# Use install if available for perms, else cp
if command -v install >/dev/null 2>&1; then
  # cp binary with executable bit
  install -m 755 "$STAGED_BIN" "$BINDIR/mlx-runner.tmp" 2>/dev/null || cp -p "$STAGED_BIN" "$BINDIR/mlx-runner.tmp"
else
  cp -p "$STAGED_BIN" "$BINDIR/mlx-runner.tmp"
fi
chmod +x "$BINDIR/mlx-runner.tmp"
mv -f "$BINDIR/mlx-runner.tmp" "$BINDIR/mlx-runner"

if [ -n "$STAGED_LIB_DIR" ] && [ -d "$STAGED_LIB_DIR" ]; then
  # Copy dylibs + metallib
  for f in "$STAGED_LIB_DIR"/*.dylib "$STAGED_LIB_DIR"/*.metallib; do
    [ -e "$f" ] || continue
    verbose "copy $f -> $LIBDIR/"
    cp -p "$f" "$LIBDIR/" 2>/dev/null || cp "$f" "$LIBDIR/"
  done
  # Also copy LICENSE/README if present (optional)
fi

# Fix rpaths for installed layout: bin -> ../lib/mlx-runner (Makefile) and fallback sibling
fix_rpaths() {
  local bin="$1" libdir="$2"
  verbose "fixing rpaths for $bin and $libdir"

  # Binary: normalize any release-hardcoded @executable_path/lib/ back to @rpath if needed
  if [ -x "$bin" ] && command -v otool >/dev/null 2>&1 && command -v install_name_tool >/dev/null 2>&1; then
    if otool -L "$bin" 2>/dev/null | grep -q "@executable_path/lib/libmlxc"; then
      verbose "normalizing binary @executable_path/lib/libmlxc -> @rpath/libmlxc"
      install_name_tool -change "@executable_path/lib/libmlxc.dylib" "@rpath/libmlxc.dylib" "$bin" 2>/dev/null || true
      install_name_tool -change "@executable_path/lib/libmlx.dylib" "@rpath/libmlx.dylib" "$bin" 2>/dev/null || true
    fi
    # Ensure rpath points to installed LIBDIR relative to BINDIR
    # Compute relative if possible: BINDIR/../lib/mlx-runner is common; just add that plus generic fallbacks
    install_name_tool -add_rpath "@executable_path/../lib/mlx-runner" "$bin" 2>/dev/null || true
    install_name_tool -add_rpath "@executable_path/lib" "$bin" 2>/dev/null || true
    # If custom LIBDIR not matching default relative, also add @executable_path computed relative? Use absolute as fallback
    # Absolute rpath is discouraged but helps custom prefixes
    if [ "$LIBDIR" != "$PREFIX/lib/mlx-runner" ] && [ "$LIBDIR" != "$PREFIX/lib/mlx-runner" ]; then
      # Try to compute relative via python or realpath if available
      rel=""
      if command -v python3 >/dev/null 2>&1; then
        rel="$(python3 -c "import os,sys; print(os.path.relpath(sys.argv[2], sys.argv[1]))" "$BINDIR" "$LIBDIR" 2>/dev/null || true)"
      elif command -v realpath >/dev/null 2>&1 && realpath --relative-to="$BINDIR" "$LIBDIR" >/dev/null 2>&1; then
        rel="$(realpath --relative-to="$BINDIR" "$LIBDIR" 2>/dev/null || true)"
      fi
      if [ -n "$rel" ] && [ "$rel" != "." ]; then
        install_name_tool -add_rpath "@executable_path/$rel" "$bin" 2>/dev/null || true
        verbose "added custom rpath @executable_path/$rel"
      fi
    fi
  elif command -v install_name_tool >/dev/null 2>&1; then
    install_name_tool -add_rpath "@executable_path/../lib/mlx-runner" "$bin" 2>/dev/null || true
    install_name_tool -add_rpath "@executable_path/lib" "$bin" 2>/dev/null || true
  fi

  # libmlxc -> libmlx
  local libmlxc="$libdir/libmlxc.dylib"
  local libmlx="$libdir/libmlx.dylib"
  if [ -f "$libmlxc" ] && command -v install_name_tool >/dev/null 2>&1; then
    if command -v otool >/dev/null 2>&1 && otool -L "$libmlxc" 2>/dev/null | grep -q "@rpath/libmlx"; then
      verbose "fixing libmlxc @rpath/libmlx -> @loader_path/libmlx"
      install_name_tool -change "@rpath/libmlx.dylib" "@loader_path/libmlx.dylib" "$libmlxc" 2>/dev/null || true
    fi
    # Also handle release path that might already be @loader_path
    install_name_tool -add_rpath "@loader_path" "$libmlxc" 2>/dev/null || true
    install_name_tool -add_rpath "@loader_path/../lib/mlx-runner" "$libmlxc" 2>/dev/null || true
  fi
  if [ -f "$libmlx" ] && command -v install_name_tool >/dev/null 2>&1; then
    install_name_tool -add_rpath "@loader_path" "$libmlx" 2>/dev/null || true
  fi
}

fix_rpaths "$BINDIR/mlx-runner" "$LIBDIR"

# Smoke test
info "verifying install"
if [ -x "$BINDIR/mlx-runner" ]; then
  if "$BINDIR/mlx-runner" --version >/dev/null 2>&1; then
    info "  $("$BINDIR/mlx-runner" --version 2>&1 | head -1)"
  else
    warn "binary --version failed"
  fi
  if "$BINDIR/mlx-runner" --metal-check >/dev/null 2>&1; then
    info "  metal-check OK"
    "$BINDIR/mlx-runner" --metal-check 2>&1 | head -n 5 | sed 's/^/  /' || true
  else
    warn "metal-check failed (binary runs but Metal not available — need macOS 26.2+ with Metal)"
    "$BINDIR/mlx-runner" --metal-check 2>&1 | head -n 10 | sed 's/^/  /' || true
  fi
  if command -v otool >/dev/null 2>&1; then
    verbose "otool -L $BINDIR/mlx-runner:"
    otool -L "$BINDIR/mlx-runner" 2>/dev/null | grep -E "libmlx|libmlxc" | sed 's/^/  /' || true
  fi
else
  err "install failed — $BINDIR/mlx-runner not executable"
  exit 1
fi

# PATH hint
case ":$PATH:" in
  *":$BINDIR:"*) ;;
  *)
    echo ""
    info "Add to PATH:"
    info "  export PATH=\"$BINDIR:\$PATH\""
    # Offer to append to shell rc if interactive
    if [ -t 0 ] && [ -t 1 ]; then
      shell_rc=""
      if [ -n "${ZSH_VERSION:-}" ] || [ "${SHELL:-}" = "/bin/zsh" ] || [ -f "$HOME/.zshrc" ]; then shell_rc="$HOME/.zshrc"
      elif [ -f "$HOME/.bashrc" ]; then shell_rc="$HOME/.bashrc"
      elif [ -f "$HOME/.bash_profile" ]; then shell_rc="$HOME/.bash_profile"
      fi
      if [ -n "$shell_rc" ]; then
        printf "[install] Append to %s? [y/N] " "$shell_rc" >&2
        read -r ans 2>/dev/null || ans="n"
        case "$ans" in
          y|Y|yes|YES) echo "export PATH=\"$BINDIR:\$PATH\"" >> "$shell_rc"; info "appended to $shell_rc (restart shell)";;
          *) info "skipped rc update";;
        esac
      fi
    fi
    ;;
esac

echo ""
info "installed $TAG to $BINDIR/mlx-runner + $LIBDIR"
info "  mlx-runner --version"
info "  mlx-runner --metal-check"
info "  mlx-runner --help"
if [ "$PREFIX" = "$PREFIX_DEFAULT" ]; then
  info "  (default prefix $PREFIX — override with --prefix /usr/local)"
fi
info "uninstall: ./install.sh --uninstall --prefix $PREFIX  (or rm $BINDIR/mlx-runner $LIBDIR)"
info "           curl -fsSL https://raw.githubusercontent.com/$REPO/main/install.sh | bash -s -- --uninstall --prefix $PREFIX"
