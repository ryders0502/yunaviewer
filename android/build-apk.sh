#!/usr/bin/env bash
# ./build-apk.sh          debug build + lint + unit tests -> dist/YunaViewer-debug.apk
# ./build-apk.sh release  minified, signed release build  -> dist/YunaViewer-release.apk
#   needs YUNA_KS_PASS (the release key's password); the key itself stays outside the repo
set -euo pipefail

cd "$(dirname "$0")"
export ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
KEYSTORE="${YUNA_KEYSTORE:-$HOME/.local/share/yunaviewer/release.jks}"

if [ "${1:-debug}" = release ]; then
    : "${YUNA_KS_PASS:?set YUNA_KS_PASS to the release key password}"
    ./gradlew --console=plain assembleRelease lintRelease testDebugUnitTest
    apksigner=$(ls "$ANDROID_HOME"/build-tools/*/apksigner | tail -1)
    mkdir -p dist
    "$apksigner" sign --ks "$KEYSTORE" --ks-key-alias yunaviewer --ks-pass env:YUNA_KS_PASS \
        --v4-signing-enabled false --out dist/YunaViewer-release.apk app/build/outputs/apk/release/app-release-unsigned.apk
    "$apksigner" verify dist/YunaViewer-release.apk
    printf 'APK: %s\n' "$PWD/dist/YunaViewer-release.apk"
else
    ./gradlew --console=plain assembleDebug lintDebug testDebugUnitTest
    install -D -m 644 app/build/outputs/apk/debug/app-debug.apk dist/YunaViewer-debug.apk
    printf 'APK: %s\n' "$PWD/dist/YunaViewer-debug.apk"
fi
