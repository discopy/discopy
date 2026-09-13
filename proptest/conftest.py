"""
Hypothesis profiles and example databases for the property suite, see
the documentation of :mod:`discopy.axioms`.

Under ``CI`` a registered profile inherits Hypothesis's ``ci`` defaults,
``derandomize=True`` and hence ``database=None``, so both are explicit.
"""

import os

from hypothesis import settings
from hypothesis.database import (
    DirectoryBasedExampleDatabase, GitHubArtifactDatabase,
    MultiplexedDatabase, ReadOnlyDatabase)

LOCAL = DirectoryBasedExampleDatabase(".hypothesis/examples")
"""
The database every run writes to: on CI it is downloaded from the
previous run's artifact before the tests and uploaded after them.
"""

COMMON = dict(
    derandomize=False, database=LOCAL, deadline=None, print_blob=True)


PROFILE = os.environ.get("HYPOTHESIS_PROFILE", "dev")

settings.register_profile("pr", max_examples=20, **COMMON)
settings.register_profile("explore", max_examples=1000, **COMMON)
settings.register_profile("dev", max_examples=100, **COMMON)
settings.register_profile("fast", max_examples=100, **COMMON)
"""
The settings of ``dev`` for a matrix of one cell per declaration of a
law, see ``once_per_declaration`` in ``test_axioms.py``.
"""
if PROFILE != "shared":
    settings.load_profile(PROFILE)


def pytest_configure(config):
    """
    Register and load the ``shared`` profile on demand: the ``dev`` budget
    over the local database backed by CI's, read-only, so that a developer
    with a ``GITHUB_TOKEN`` replays what CI found without recording
    anything. Building the artifact database touches storage, which
    Hypothesis warns against at conftest import, so it happens only when
    the profile is asked for.
    """
    if PROFILE == "shared":
        database = MultiplexedDatabase(LOCAL, ReadOnlyDatabase(
            GitHubArtifactDatabase("discopy", "discopy")))
        settings.register_profile(
            "shared", max_examples=100, **dict(COMMON, database=database))
        settings.load_profile("shared")
