#!/usr/bin/env bash
set -euo pipefail

# Build a Debian/Ubuntu .deb package for ZoidCraft (amd64).
# Usage:
#   packaging/linux/build_deb.sh <zoidcraft-bin> <zoidcraft-cli-bin> <version> <output-deb>

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <zoidcraft-bin> <zoidcraft-cli-bin> <version> <output-deb>" >&2
    exit 1
fi

GUI_BIN="$1"
CLI_BIN="$2"
VERSION="${3#v}"
OUT_DEB="$4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/zoidcraft-deb.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

PKG_ROOT="${WORK_DIR}/pkg"
mkdir -p \
    "${PKG_ROOT}/DEBIAN" \
    "${PKG_ROOT}/usr/bin" \
    "${PKG_ROOT}/usr/share/applications" \
    "${PKG_ROOT}/usr/share/icons/hicolor/256x256/apps" \
    "${PKG_ROOT}/usr/share/icons/hicolor/512x512/apps" \
    "${PKG_ROOT}/usr/share/doc/zoidcraft"

cp "${GUI_BIN}" "${PKG_ROOT}/usr/bin/zoidcraft"
cp "${CLI_BIN}" "${PKG_ROOT}/usr/bin/zoidcraft-cli"
chmod 755 "${PKG_ROOT}/usr/bin/zoidcraft" "${PKG_ROOT}/usr/bin/zoidcraft-cli"

cp "${SCRIPT_DIR}/zoidcraft.desktop" "${PKG_ROOT}/usr/share/applications/zoidcraft.desktop"
cp "${REPO_ROOT}/packaging/icons/zoidcraft-256.png" "${PKG_ROOT}/usr/share/icons/hicolor/256x256/apps/zoidcraft.png"
cp "${REPO_ROOT}/packaging/icons/zoidcraft-512.png" "${PKG_ROOT}/usr/share/icons/hicolor/512x512/apps/zoidcraft.png"
cp "${REPO_ROOT}/LICENSE" "${PKG_ROOT}/usr/share/doc/zoidcraft/LICENSE"
cp "${REPO_ROOT}/README.md" "${PKG_ROOT}/usr/share/doc/zoidcraft/README.md"
cp "${REPO_ROOT}/THIRD_PARTY_LICENSES.md" "${PKG_ROOT}/usr/share/doc/zoidcraft/THIRD_PARTY_LICENSES.md"

INSTALLED_KB="$(du -sk "${PKG_ROOT}/usr" | awk '{print $1}')"

cat > "${PKG_ROOT}/DEBIAN/control" <<EOF
Package: zoidcraft
Version: ${VERSION}
Section: video
Priority: optional
Architecture: amd64
Installed-Size: ${INSTALLED_KB}
Depends: libc6, libasound2 | libasound2t64, libgtk-3-0 | libgtk-3-0t64, libxkbcommon0
Recommends: ffmpeg
Maintainer: ZoidCraft Contributors <https://github.com/rbstxt/zoidcraft>
Homepage: https://github.com/rbstxt/zoidcraft
Description: GPU-accelerated 3D/2D motion-graphics editor and CLI renderer
 ZoidCraft is a native Rust + wgpu motion-graphics editor supporting 3D scenes,
 2D compositing, 80+ GPU effects, PanZoid (.pz) interop, and FFmpeg export.
EOF

mkdir -p "$(dirname "${OUT_DEB}")"
dpkg-deb --root-owner-group --build "${PKG_ROOT}" "${OUT_DEB}"
dpkg-deb --info "${OUT_DEB}"
echo "Built Debian package: ${OUT_DEB}"
