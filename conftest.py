"""Root conftest: fallback benchmark fixture for micro-benchmarks in test/*/.

When pytest-codspeed is installed (uv sync --extra bench), its plugin
provides the real `benchmark` fixture with timing instrumentation.
When it is not installed, this fallback makes the fixture available as a
no-op so that benchmark functions collected during normal `uv run pytest`
runs pass without error.
"""
import pytest

try:
    import pytest_codspeed  # noqa: F401 — plugin registered; it owns `benchmark`
except ImportError:
    @pytest.fixture
    def benchmark():
        def _passthrough(fn):
            return fn()
        return _passthrough
