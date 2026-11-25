#!/usr/bin/env bash
set -euo pipefail

# Build a ready-to-use signed macOS ZoidCraft.app bundle and compressed .dmg installer.
# Usage:
#   packaging/macos/build_dmg.sh <zoidcraft-bin> <zoidcraft-cli-bin> <version> <output-dmg>

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <zoidcraft-bin> <zoidcraft-cli-bin> <version> <output-dmg>" >&2
    exit 1
fi

GUI_BIN="$1"
CLI_BIN="$2"
VERSION="${3#v}"
OUT_DMG="$4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/zoidcraft-dmg.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

DMG_STAGE="${WORK_DIR}/dmg-root"
APP_BUNDLE="${DMG_STAGE}/ZoidCraft.app"

mkdir -p \
    "${APP_BUNDLE}/Contents/MacOS" \
    "${APP_BUNDLE}/Contents/Resources"

cp "${GUI_BIN}" "${APP_BUNDLE}/Contents/MacOS/zoidcraft"
cp "${CLI_BIN}" "${APP_BUNDLE}/Contents/MacOS/zoidcraft-cli"
chmod 755 "${APP_BUNDLE}/Contents/MacOS/zoidcraft" "${APP_BUNDLE}/Contents/MacOS/zoidcraft-cli"

printf 'APPL????' > "${APP_BUNDLE}/Contents/PkgInfo"
sed "s/__VERSION__/${VERSION}/g" "${SCRIPT_DIR}/Info.plist.template" > "${APP_BUNDLE}/Contents/Info.plist"

cp "${SCRIPT_DIR}/AppIcon.icns" "${APP_BUNDLE}/Contents/Resources/AppIcon.icns"
cp "${REPO_ROOT}/LICENSE" "${APP_BUNDLE}/Contents/Resources/LICENSE"
cp "${REPO_ROOT}/README.md" "${APP_BUNDLE}/Contents/Resources/README.md"
cp "${REPO_ROOT}/THIRD_PARTY_LICENSES.md" "${APP_BUNDLE}/Contents/Resources/THIRD_PARTY_LICENSES.md"

# Ad-hoc code sign the bundle so Apple Silicon macOS launches cleanly without Mach-O signature errors.
codesign --force --deep --sign - "${APP_BUNDLE}"
codesign --verify --deep --strict "${APP_BUNDLE}"

# Standard macOS drag-to-Applications symlink inside the DMG.
ln -s /Applications "${DMG_STAGE}/Applications"

mkdir -p "$(dirname "${OUT_DMG}")"
rm -f "${OUT_DMG}"

hdiutil create \
    -volname "ZoidCraft ${VERSION}" \
    -srcfolder "${DMG_STAGE}" \
    -ov \
    -format UDZO \
    -imagekey zlib-level=9 \
    "${OUT_DMG}"

hdiutil verify "${OUT_DMG}"
echo "Built and verified macOS DMG: ${OUT_DMG}"
