#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/build"
swiftc -O \
  -framework AVFoundation \
  -framework Vision \
  -framework CoreGraphics \
  -framework CoreMedia \
  -framework CoreVideo \
  "$ROOT/macos/gaze_bridge.swift" \
  -o "$ROOT/build/gaze_bridge"
echo "Built: $ROOT/build/gaze_bridge"
"$ROOT/build/gaze_bridge" --self-test
