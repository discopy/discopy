"""Tests for the composite action the build jobs share.

A composite step takes a strict subset of what a workflow step takes, and
the runner only says so when the job starts: `timeout-minutes` on one of
these cost a whole CI round.
"""

import pathlib

import pytest
import yaml

GITHUB = pathlib.Path(__file__).resolve().parent.parent
STEP = {"name", "id", "if", "uses", "with", "run", "shell", "env",
        "working-directory", "continue-on-error"}


@pytest.fixture(scope="module")
def action():
    with open(GITHUB / "actions" / "setup" / "action.yml") as file:
        return yaml.safe_load(file)


def test_every_step_key_is_one_a_composite_step_takes(action):
    for step in action["runs"]["steps"]:
        assert set(step) <= STEP, f"{set(step) - STEP} in {step.get('name')}"


def test_every_script_names_its_shell(action):
    for step in action["runs"]["steps"]:
        assert "run" not in step or step.get("shell"), step.get("name")


def test_every_input_the_workflows_pass_is_declared(action):
    passed = set()
    for path in (GITHUB / "workflows").glob("*.yml"):
        with open(path) as file:
            workflow = yaml.safe_load(file)
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("./.github/actions/setup"):
                    passed |= set(step.get("with", {}))
    assert passed and passed <= set(action["inputs"])
