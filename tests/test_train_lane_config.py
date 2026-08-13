"""Check herdmux.train.json against the code it claims to drive.

The GPU lane reads this file to decide what to run. Nothing else in the repo
parses it, so a typo -- in the JSON itself, in an entrypoint name, in a flag or
an env var that has since been renamed -- would surface as a failed container
run an hour into the lane, or worse as a run that starts and measures the wrong
thing. These tests are cheap and run on CPU; the lane is not.
"""

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "herdmux.train.json"

# The entrypoints the lane runs, in the order it runs them.
LANE_SOURCES = ("prepare.py", "train.py")

# `import a.b as c` / `from a.b import c` -> the root package pip must supply.
IMPORT_RE = re.compile(r"^[ \t]*(?:import|from)[ \t]+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)

# A quoted requirement inside the command string, e.g. "pyarrow>=21.0.0".
REQUIREMENT_RE = re.compile(r'"([A-Za-z0-9_.\-]+)(>=[0-9][0-9A-Za-z.\-]*)"')


def installed_requirements(config):
    """{package: '>=version'} for every quoted specifier in the pip install step."""
    return dict(REQUIREMENT_RE.findall(config["command"]))


def third_party_imports():
    """Root packages imported by the lane's entrypoints that pip must provide."""
    local_modules = {path.stem for path in REPO_ROOT.glob("*.py")}
    roots = set()
    for name in LANE_SOURCES:
        roots.update(IMPORT_RE.findall((REPO_ROOT / name).read_text(encoding="utf-8")))
    return roots - sys.stdlib_module_names - local_modules


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


def test_install_step_covers_every_import_the_entrypoints_make(config):
    # The expensive failure this guards: an import added to train.py alone gets
    # past the install step and past prepare.py, and only raises after the ~1 GB
    # parquet download and the BPE tokenizer training have been paid for --
    # an hour of GPU lane spent to reach an ImportError.
    supplied = set(installed_requirements(config)) | {"torch"}  # torch comes from the image
    missing = third_party_imports() - supplied
    assert not missing, f"the lane's entrypoints import {sorted(missing)}, which pip never installs"


def test_install_step_does_not_fight_the_image_over_torch(config):
    # pyproject pins torch from the cu128 index; the lane runs on a cuda124
    # image that already ships a matching build. Installing torch here would
    # swap the image's working build for one compiled against another CUDA.
    # Checked against the raw install step, not installed_requirements(): a
    # bare unquoted `torch` is exactly the regression this guards, and it is
    # invisible to a parser that only sees quoted version specifiers.
    install_step = config["command"].split("&&")[0]
    assert not re.search(r"\btorch\b", install_step), "the image supplies torch; do not reinstall it"


def test_version_floors_match_pyproject(config):
    # A bare `pip install pyarrow` is satisfied by whatever the image already
    # ships, however old -- so the floors are what actually force prepare.py to
    # run at or above the version this repo declares it needs.
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = installed_requirements(config)
    assert requirements, "the install step must pin floors, not bare package names"
    for package, floor in requirements.items():
        assert f"{package}{floor}" in pyproject, (
            f"the lane installs {package}{floor}, which pyproject.toml does not declare"
        )


def test_requirements_are_quoted_against_shell_redirection(config):
    # The runner executes this string under `bash -lc`. An unquoted
    # pyarrow>=21.0.0 is `pip install pyarrow` with stdout redirected into a
    # file named `=21.0.0` -- the unbounded install the floors exist to
    # prevent, plus junk written into the worktree, and exit code 0 either way.
    command = config["command"]
    for specifier in REQUIREMENT_RE.finditer(command):
        command = command.replace(specifier.group(0), "")
    assert ">" not in command, "version specifiers must be double-quoted inside the command"


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
