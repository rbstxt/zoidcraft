#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:$PATH"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$root/target}"
staging="$(mktemp -d "${TMPDIR:-/tmp}/zoidcraft-web.XXXXXX")"
trap 'rm -rf "$staging"' EXIT
mkdir -p "$staging/pkg" "$staging/icons" "$staging/vendor"

CARGO_PROFILE_RELEASE_STRIP=debuginfo cargo build --locked --target wasm32-unknown-unknown --release --lib -p zoidcraft
wasm-bindgen --target web --no-typescript --remove-name-section --out-dir "$staging/pkg" \
    "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/zoidcraft.wasm"
wasm-opt -O2 --enable-bulk-memory --enable-reference-types --enable-multivalue \
    --enable-sign-ext --enable-nontrapping-float-to-int \
    "$staging/pkg/zoidcraft_bg.wasm" -o "$staging/pkg/optimized.wasm"
mv "$staging/pkg/optimized.wasm" "$staging/pkg/zoidcraft_bg.wasm"
node --input-type=module -e '
    import { readFileSync } from "node:fs";
    new WebAssembly.Module(readFileSync(process.argv[1]));
' "$staging/pkg/zoidcraft_bg.wasm"

npm --prefix web ci --no-audit --no-fund
node web/bundle.mjs "$staging/vendor"
cp web/zc_web_runtime.js web/_headers "$staging/"
cp packaging/icons/zoidcraft-256.png "$staging/icons/"
cp THIRD_PARTY_LICENSES.md "$staging/"
for module in mp4box mp4-muxer webm-muxer fflate; do
    cp "web/node_modules/$module/LICENSE" "$staging/vendor/$module.LICENSE"
done

# The shell revalidates on navigation. Versioned URLs bust the immutable asset cache.
build_id="$(shasum -a 256 web/index.html "$staging"/pkg/* "$staging"/vendor/*.js \
    "$staging"/icons/* "$staging/zc_web_runtime.js" | awk '{print $1}' | shasum -a 256 | cut -c1-16)"
sed "s/__ZC_BUILD_ID__/$build_id/g" web/index.html > "$staging/index.html"
rm -rf web/dist
mv "$staging" web/dist

wasm=web/dist/pkg/zoidcraft_bg.wasm
printf 'WebAssembly: %s bytes raw, %s bytes gzipped\n' \
    "$(wc -c < "$wasm" | tr -d ' ')" "$(gzip -n -9 -c "$wasm" | wc -c | tr -d ' ')"
