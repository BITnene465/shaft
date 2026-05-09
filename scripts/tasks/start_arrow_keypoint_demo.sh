#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# The demo talks to local vLLM. Clearing proxy variables avoids Gradio/httpx
# SOCKS import issues in this environment.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-127.0.0.1}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7861}"

exec .venv/bin/python scripts/tasks/arrow_keypoint_demo.py
