#!/usr/bin/env bash
set -euo pipefail

ROOT="{{repo_root}}"

# shellcheck disable=SC1091
. "{{env_script}}"

exec "$@"
