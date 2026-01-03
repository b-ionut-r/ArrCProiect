#!/usr/bin/bash

set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
COMPILE_COMMANDS="${BUILD_DIR}/compile_commands.json"

if [[ ! -f "${COMPILE_COMMANDS}" ]]; then
    echo "cppcheck: compile_commands.json not found; skipping."
    exit 0
fi

if ! grep -Eq '\\.(c|cc|cxx|cpp)"' "${COMPILE_COMMANDS}"; then
    echo "cppcheck: no C/C++ sources found; skipping."
    exit 0
fi

bash ./scripts/run_cppcheck.sh
