"""Tests for the style-review workflow's own YAML, not its Python scripts."""

import pathlib

import yaml

GITHUB = pathlib.Path(__file__).resolve().parent.parent


def test_concurrency_group_isolates_a_non_review_comment():
    """`issue_comment` fires this workflow for every comment on the pull
    request, not just the ones asking for a review, and GitHub cancels
    the older run of a shared group before either job's own `if` can
    skip the no-op ones. A trigger this job would not review must fall
    back to a group keyed by its own run, so it can never collide with —
    and cancel — a real round in progress."""
    with open(GITHUB / "workflows" / "style-review.yml") as file:
        workflow = yaml.safe_load(file)
    group = workflow["concurrency"]["group"]
    assert "github.run_id" in group
    assert "'review'" in group
