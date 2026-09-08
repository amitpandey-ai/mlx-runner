PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin
LIBDIR ?= $(PREFIX)/lib/mlx-runner
ZIG ?= $(CURDIR)/.zig-toolchain/zig
MLX_STAGE ?= $(CURDIR)/lib/mlx

# Fallback to system zig if .zig-toolchain not present
ifeq (,$(wildcard $(ZIG)))
ZIG := zig
endif

.PHONY: all build lib fetch-zig test test-mlx metal-check install uninstall clean help

all: build

help:
	@echo "mlx-runner — Zig 0.17, mlx 0.32.2 (Qwen3.5 GDN)"
	@echo "  make              - fetch Zig, build lib/mlx, zig build -Doptimize=ReleaseFast"
	@echo "  make build        - same as make"
	@echo "  make lib          - only bash scripts/build-mlx.sh"
	@echo "  make test         - zig build test --summary all (hermetic, 31 tests)"
	@echo "  make test-mlx     - zig build test-mlx (needs libmlx, Metal)"
	@echo "  make metal-check  - ./zig-out/bin/mlx-runner --metal-check"
	@echo "  make install      - build + install to \$$(BINDIR) [PREFIX=$(PREFIX)]"
	@echo "  make uninstall    - remove \$$(BINDIR)/mlx-runner and \$$(LIBDIR)"
	@echo "  make clean        - rm -rf zig-out .zig-cache"

fetch-zig:
	@bash scripts/fetch-zig.sh

lib:
	@bash scripts/build-mlx.sh
	@ls -lh $(MLX_STAGE)/lib/libmlx.dylib $(MLX_STAGE)/lib/libmlxc.dylib $(MLX_STAGE)/lib/mlx.metallib 2>/dev/null | awk '{print $$9, $$5}'

build: fetch-zig lib
	@$(ZIG) build -Doptimize=ReleaseFast
	@ls -lh zig-out/bin/mlx-runner
	@otool -L zig-out/bin/mlx-runner | grep -E "libmlx|libmlxc" || true

test:
	@$(ZIG) build test --summary all

test-mlx:
	@$(ZIG) build test-mlx --summary all

metal-check: build
	@./zig-out/bin/mlx-runner --metal-check

install: build
	@mkdir -p $(BINDIR) $(LIBDIR)
	@cp zig-out/bin/mlx-runner $(BINDIR)/mlx-runner
	@cp $(MLX_STAGE)/lib/libmlx.dylib $(LIBDIR)/ 2>/dev/null || true
	@cp $(MLX_STAGE)/lib/libmlxc.dylib $(LIBDIR)/ 2>/dev/null || true
	@cp $(MLX_STAGE)/lib/libjaccl.dylib $(LIBDIR)/ 2>/dev/null || true
	@cp $(MLX_STAGE)/lib/mlx.metallib $(LIBDIR)/ 2>/dev/null || true
	@chmod +x $(BINDIR)/mlx-runner
	@# Fix rpaths for installed layout: bin -> ../lib/mlx-runner
	@install_name_tool -add_rpath "@executable_path/../lib/mlx-runner" $(BINDIR)/mlx-runner 2>/dev/null || true
	@install_name_tool -add_rpath "@loader_path/../lib/mlx-runner" $(LIBDIR)/libmlxc.dylib 2>/dev/null || true
	@install_name_tool -add_rpath "@loader_path" $(LIBDIR)/libmlxc.dylib 2>/dev/null || true
	@install_name_tool -change @rpath/libmlx.dylib @loader_path/libmlx.dylib $(LIBDIR)/libmlxc.dylib 2>/dev/null || true
	@echo "Installed $(BINDIR)/mlx-runner + $(LIBDIR)/{libmlx,libmlxc,mlx.metallib}"
	@$(BINDIR)/mlx-runner --version
	@$(BINDIR)/mlx-runner --metal-check 2>&1 | head -n 5
	@echo "Tip: ensure $(BINDIR) is in PATH: export PATH=\"$(BINDIR):\$$PATH\""

uninstall:
	@rm -f $(BINDIR)/mlx-runner
	@rm -rf $(LIBDIR)
	@echo "Uninstalled $(BINDIR)/mlx-runner and $(LIBDIR)"

clean:
	@rm -rf zig-out .zig-cache lib/.mlx-build
	@echo "Cleaned zig-out .zig-cache"
