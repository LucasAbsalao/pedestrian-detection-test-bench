#!/usr/bin/env bash
set -euo pipefail

# Install dependencies and keep opencv-contrib-python as the ONLY cv2 provider.
#
# Why this is needed:
#   - ultralytics  depends on opencv-python
#   - label-studio depends on opencv-python-headless (via label-studio-sdk)
#   - this project wants opencv-contrib-python
#
# All three packages ship the same `cv2` module and overwrite each other's files
# in site-packages, so Poetry (which resolves by package name, not module name)
# happily installs all three. We remove the two we don't want right after install.
poetry install

poetry run python -m pip uninstall -y opencv-python opencv-python-headless

echo ""
echo "Done. opencv-contrib-python is now the sole cv2 provider."
poetry run python -m pip list | grep -i opencv || true
