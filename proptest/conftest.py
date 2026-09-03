"""
Hypothesis profiles and example databases for the property suite, see
PROPTEST.md.

Under ``CI`` a registered profile inherits Hypothesis's ``ci`` defaults,
``derandomize=True`` and hence ``database=None``, so both are explicit.
"""

import os

from hypothesis import HealthCheck, settings
from hypothesis.database import (
    DirectoryBasedExampleDatabase, GitHubArtifactDatabase,
    MultiplexedDatabase, ReadOnlyDatabase)

LOCAL = DirectoryBasedExampleDatabase(".hypothesis/examples")
"""
The database every run writes to: on CI it is downloaded from the
previous run's artifact before the tests and uploaded after them.
"""

COMMON = dict(
    derandomize=False, database=LOCAL, deadline=None, print_blob=True,
    suppress_health_check=[HealthCheck.filter_too_much])


def shared() -> MultiplexedDatabase:
    """
    The local database backed by CI's, read-only, so that a developer with
    a ``GITHUB_TOKEN`` replays what CI found without recording anything.
    """
    return MultiplexedDatabase(LOCAL, ReadOnlyDatabase(
        GitHubArtifactDatabase("discopy", "discopy")))


settings.register_profile("pr", max_examples=20, **COMMON)
settings.register_profile("explore", max_examples=1000, **COMMON)
settings.register_profile("dev", max_examples=100, **dict(
    COMMON, database=shared() if "GITHUB_TOKEN" in os.environ else LOCAL))
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


def pytest_runtest_setup(item):
    """
    Key the database of a Hypothesis cell by its node id.

    Hypothesis keys the database by the digest of the test function, which
    every parameter of a parametrized test shares, so one cell's failures
    would be replayed against every other cell.
    """
    if hasattr(item.obj, "hypothesis"):
        item.obj._hypothesis_internal_database_key = item.nodeid.encode()
