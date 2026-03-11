#!/bin/bash
# Ferrite-OS Daemon Launcher
#
# This script sets up the environment and runs the daemon.

# Resolve symlinks to find the real script location
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    # If SOURCE is relative, resolve it relative to the symlink's directory
    [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

# Set library path
export LD_LIBRARY_PATH="${SCRIPT_DIR}/lib:${LD_LIBRARY_PATH}"

# Detect GPU name if not already set
if [ -z "$FERRITE_GPU_NAME" ] && command -v nvidia-smi &>/dev/null; then
    export FERRITE_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
fi

# Run daemon
exec "${SCRIPT_DIR}/target/release/ferrite-daemon" "$@"
