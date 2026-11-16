#!/usr/bin/env bash
# Starts n8n bound to localhost only. Holds no secrets: the Anthropic key
# lives in an n8n credential, not in the environment.
set -euo pipefail

# Default is '::' (every interface). Keep uploads and the editor off the LAN.
export N8N_LISTEN_ADDRESS=127.0.0.1

# Already the default in regular mode; pinned so uploads never sit in memory.
export N8N_DEFAULT_BINARY_DATA_MODE=filesystem

# Per-file upload cap for the chat widget, in MiB (n8n default: 200).
export N8N_FORMDATA_FILE_SIZE_MAX=50

exec n8n start
