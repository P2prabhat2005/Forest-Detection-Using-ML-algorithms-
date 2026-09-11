#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the Forest Fire Detection project.
#
# Creates a project-local virtual environment and installs the pinned
# Python dependencies from requirements.txt.
#
# Note: the base image's `python3-venv` package does not ship `ensurepip`,
# and Debian/Ubuntu apt mirrors are not reachable from the agent network,
# so we create the venv with `--without-pip` and bootstrap pip from PyPI
# (which is reachable) via the official get-pip.py installer.
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}" --without-pip
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if ! python -m pip --version >/dev/null 2>&1; then
  curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  python /tmp/get-pip.py
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Environment ready. Activate with: source ${VENV_DIR}/bin/activate"
