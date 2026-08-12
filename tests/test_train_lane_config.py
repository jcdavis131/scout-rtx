"""Check herdmux.train.json against the code it claims to drive.

The GPU lane reads this file to decide what to run. Nothing else in the repo
parses it, so a typo -- in the JSON itself, in an entrypoint name, in a flag or
an env var that has since been renamed -- would surface as a failed container
run an hour into the lane, or worse as a run that starts and measures the wrong
thing. These tests are cheap and run on CPU; the lane is not.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "herdmux.train.json"


@pytest.fixture(scope="module")
def config():
    # Also the parse check: invalid JSON fails every test in this module.
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_lane_config_has_the_fields_the_runner_needs(config):
    for key in ("command", "readOnly", "env", "timeoutMs"):
        assert key in config, f"herdmux.train.json is missing {key!r}"
    assert isinstance(config["command"], str) and config["command"].strip()
    assert isinstance(config["timeoutMs"], int) and config["timeoutMs"] > 0


def test_command_runs_entrypoints_that_exist(config):
    for entrypoint in ("prepare.py", "train.py"):
        assert entrypoint in config["command"]
        assert (REPO_ROOT / entrypoint).is_file()


def test_command_only_uses_flags_train_py_declares(config):
    # The small-run knob. If train.py ever renames it, the lane would run a full
    # TIME_BUDGET training instead of a smoke test -- the exact overrun this
    # config's timeoutMs is not sized for.
    assert "--smoke-test" in config["command"]
    assert '"--smoke-test"' in (REPO_ROOT / "train.py").read_text(encoding="utf-8")


def test_env_vars_are_ones_the_code_actually_reads(config):
    sources = "\n".join(
        (REPO_ROOT / name).read_text(encoding="utf-8") for name in ("train.py", "prepare.py")
    )
    assert config["env"], "env is what keeps artifacts out of the checked-out repo"
    for name in config["env"]:
        assert name in sources, f"{name} is set for the lane but read by nothing"


def test_read_only_paths_exist_and_hold_committed_state(config):
    # A readOnly entry that no longer exists silently protects nothing.
    assert config["readOnly"], "committed data must be mounted read-only"
    for entry in config["readOnly"]:
        path = REPO_ROOT / entry
        assert path.is_dir(), f"readOnly entry {entry!r} is not a directory in this repo"
        assert any(path.iterdir()), f"readOnly entry {entry!r} is empty"


def test_out_dir_is_outside_the_worktree(config):
    # A measuring run must not leave a checkpoint in the published worktree.
    # This repo ships no weights, but that is the accident this guards against.
    out_dir = config["env"]["AUTORESEARCH_OUT_DIR"]
    assert out_dir.startswith("/"), "AUTORESEARCH_OUT_DIR must be an absolute mount path"
    assert not out_dir.startswith("/work"), "artifacts must not land in the checked-out repo"
