#!/bin/bash
# Install the full development environment in a Claude Code on the web
# session, so that `uv run pflake8 discopy` and `uv run coverage run -m pytest`
# run as CONTRIBUTING.md says, extras included.
#
# `pyproject.toml` pins torch to the CPU index at download.pytorch.org on
# Linux. A session whose network policy does not allow that host cannot sync
# the `quantum` extra, and every torch test used to be skipped: the fallback
# syncs everything but torch and installs the locked version from PyPI, which
# ships CUDA wheels that run on the CPU. The allowlist is the better fix.
set -uo pipefail

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

log() { echo "session-start: $*" >&2; }
cd "${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 0

if uv sync --dev --group all >/dev/null 2>&1; then
  log "uv sync --dev --group all"
  exit 0
fi

log "uv sync --dev --group all failed, syncing without torch"
if ! uv sync --dev --group all --no-install-package torch >/dev/null 2>&1; then
  log "uv sync failed: run 'uv sync --dev' by hand"
  exit 0
fi

version="$(awk '/^name = "torch"$/ { found = 1 } found && /^version/ { print $3; exit }' uv.lock | tr -d '"')"
if uv pip install --index-url https://pypi.org/simple "torch==$version" >/dev/null 2>&1; then
  log "torch $version installed from PyPI instead of download.pytorch.org"
else
  log "torch unavailable: run 'uv run pytest --skip-extra'"
fi
exit 0
