"""Tests for the style-review workflow's own YAML, not its Python scripts."""

import pathlib

import yaml

GITHUB = pathlib.Path(__file__).resolve().parent.parent


def test_concurrency_group_is_keyed_by_event_kind_too():
    """`issue_comment` fires this workflow for every comment on the pull
    request, not just the ones asking for a review, and GitHub cancels
    the older run of a shared group before either job's own `if` can
    skip the no-op ones — so a stray comment must not share a group
    with a push's round, or vice versa."""
    with open(GITHUB / "workflows" / "style-review.yml") as file:
        workflow = yaml.safe_load(file)
    assert "github.event_name" in workflow["concurrency"]["group"]
