#!/usr/bin/env bash
# Пример установки на LXC: venv, копирование файлов, systemd.
# Запуск: sudo bash install-lxc.sh
set -euo pipefail
DEST="${DEST:-/opt/openclaw-bridge}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"

echo "Installing from $SRC to $DEST"
mkdir -p "$DEST"
cp -a "$SRC/app.py" "$SRC/tool_parse.py" "$SRC/requirements.txt" "$DEST/"
python3 -m venv "$DEST/venv"
"$DEST/venv/bin/pip" install -U pip
"$DEST/venv/bin/pip" install -r "$DEST/requirements.txt"

echo "Copy systemd unit (adjust paths if needed):"
echo "  sudo cp $SRC/deploy/openclaw-bridge.service /etc/systemd/system/"
echo "  sudo cp $SRC/deploy/env.example /etc/openclaw-bridge.env"
echo "  sudo chmod 600 /etc/openclaw-bridge.env"
echo "  # отредактируйте UPSTREAM_BASE_URL и при необходимости порт"
echo "  sudo systemctl daemon-reload && sudo systemctl enable --now openclaw-bridge"
