"""Contract tests for herdmux.train.json against the code it actually runs.

`herdmux.train.json` is what opts this repo into the GPU lane: a runner reads it
and launches `command` in a container. Nothing in the repo referenced it, so it
could drift out of sync silently -- rename `--smoke-test`, drop an import, or
typo `readOnly` and the config still looks fine. The failure then surfaces an
hour later as a container that died in the install step, or worse, as a run that
looked healthy while writing into a directory it was supposed to be locked out
of.

These tests hold the config to the source. They are string-and-AST checks on
files, so they need neither a GPU nor the packages the container installs.
"""

import ast
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "herdmux.train.json"

# Keys the herdmux runner understands (docker/herdmux.train.example.json).
# Anything else is either a `$`-prefixed prose comment or a typo -- and a typo
# is silently ignored by the runner, which is the dangerous case.
SCHEMA_KEYS = {"command", "image", "readOnly", "timeoutMs", "shmSize", "memory", "env"}

# The image supplies torch; installing it would fight the image's CUDA build.
IMAGE_PROVIDED = {"torch"}

SIZE_UNITS = {"k": 1 / (1024 * 1024), "m": 1 / 1024, "g": 1.0}

# docs/HARDWARE_PROFILE.md: the Docker VM has ~7.75 GB of RAM total.
VM_MEMORY_GB = 7.75


# utf-8-sig, not utf-8, wherever the text is handed to a parser: a BOM survives
# a plain utf-8 decode as a leading U+FEFF, which makes both json.loads and
# ast.parse raise. It is a no-op on BOM-less files.
PARSE_ENCODING = "utf-8-sig"


@pytest.fixture(scope="module")
def config():
    return json.loads(CONFIG_PATH.read_text(encoding=PARSE_ENCODING))


# --- helpers ---------------------------------------------------------------


def _segments(command):
    """The `&&`-joined stages of the shell command, each as a token list."""
    return [part.split() for part in command.split("&&")]


def _script_and_flags(segment):
    """('train.py', ['--smoke-test']) for a segment, or (None, []).

    Flags before the script name belong to the interpreter (`python -u`), not
    to the script, so only tokens after the `.py` token are collected.
    """
    for i, token in enumerate(segment):
        if token.endswith(".py"):
            return token, [t for t in segment[i + 1:] if t.startswith("-")]
    return None, []


def _declared_flags(source):
    """Every flag string passed to an add_argument() call in `source`."""
    return set(re.findall(r"""add_argument\(\s*["'](--[^"']+)["']""", source))


def _module_level_import_roots(path):
    """Top-level package names imported at module scope in `path`.

    Module scope only: a function-local import is paid lazily and need not be
    installed for the module to load, so it is not part of the install
    contract. `import pyarrow.parquet` counts as `pyarrow`.
    """
    tree = ast.parse(path.read_text(encoding=PARSE_ENCODING))
    roots = set()
    for node in tree.body:  # body, not walk -- module scope only
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _third_party(roots):
    """Drop the stdlib and this repo's own root-level modules."""
    local = {p.stem for p in REPO_ROOT.glob("*.py")}
    return {r for r in roots if r not in sys.stdlib_module_names and r not in local}


def _parse_size_gb(value):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmg])", value.strip().lower())
    assert match, f"unparseable docker size {value!r}"
    return float(match.group(1)) * SIZE_UNITS[match.group(2)]


# --- the file itself -------------------------------------------------------


def test_config_exists_and_parses():
    # Without this file the GPU lane does not run at all, so nothing measured
    # on real hardware is possible. The `$`-prefixed prose keys are ordinary
    # JSON strings, not comments, so the file must still be strict JSON.
    assert CONFIG_PATH.exists()
    assert isinstance(json.loads(CONFIG_PATH.read_text(encoding=PARSE_ENCODING)), dict)


def test_only_schema_keys_and_prose_keys_are_present(config):
    # A typo'd key (`readonly`, `timeoutMS`) is not an error to the runner --
    # it is silently ignored, so the protection it was meant to buy just
    # quietly does not happen.
    real_keys = {k for k in config if not k.startswith("$")}
    assert real_keys <= SCHEMA_KEYS, f"unknown key(s): {sorted(real_keys - SCHEMA_KEYS)}"


def test_command_is_unbuffered(config):
    # With buffering on, an empty log is indistinguishable from a job that
    # never started -- the exact ambiguity `python -u` removes.
    for segment in _segments(config["command"]):
        if segment and segment[0] == "python":
            assert "-u" in segment, f"buffered python invocation: {' '.join(segment)}"


# --- the command matches the entrypoints -----------------------------------


def test_every_script_in_the_command_exists(config):
    scripts = [s for s, _ in map(_script_and_flags, _segments(config["command"])) if s]
    assert scripts, "command runs no python script"
    for script in scripts:
        assert (REPO_ROOT / script).is_file(), f"{script} is not in the repo"


def test_every_flag_the_command_passes_is_declared_by_its_script(config):
    # This is the drift that costs a whole GPU cycle: rename or drop
    # `--smoke-test` and the container runs the *full* training job under a
    # timeout instead of the 3-step smoke path.
    checked = 0
    for segment in _segments(config["command"]):
        script, flags = _script_and_flags(segment)
        if script is None:
            continue
        declared = _declared_flags((REPO_ROOT / script).read_text(encoding="utf-8"))
        for flag in flags:
            assert flag in declared, f"{script} does not declare {flag}"
            checked += 1
    assert checked, "no script flags were checked -- the parser found nothing"


def test_the_smoke_path_is_what_gets_run(config):
    # The first version of this config is deliberately small: its job is to
    # prove the lane runs, not to train a shippable model.
    assert "--smoke-test" in config["command"]


def test_pip_specifiers_cannot_be_read_as_shell_redirection(config):
    # This has bitten this file before: `pip install pyarrow>=21.0.0` runs under
    # `bash -lc`, so `>=21.0.0` is a redirect and the container silently creates
    # an empty file named `=21.0.0` in the worktree instead of pinning anything.
    # Four such files were committed by a cycle that did exactly this.
    install = next(s for s in _segments(config["command"]) if s[:2] == ["pip", "install"])
    unquoted = re.sub(r"'[^']*'|\"[^\"]*\"", "", " ".join(install))
    assert not set("<>") & set(unquoted), f"unquoted redirection in: {' '.join(install)}"


def test_pip_installs_exactly_the_imports_the_image_does_not_supply(config):
    # Under-install and the container dies in the install step; over-install
    # and pip may drag in a torch that fights the image's CUDA build.
    install = next(s for s in _segments(config["command"]) if s[:2] == ["pip", "install"])
    installed = {t for t in install[2:] if not t.startswith("-")}

    needed = set()
    for script in ("prepare.py", "train.py"):
        needed |= _third_party(_module_level_import_roots(REPO_ROOT / script))

    assert installed == needed - IMAGE_PROVIDED
    assert not (installed & IMAGE_PROVIDED), "the image supplies torch; do not install it"


# --- environment and artifact paths ----------------------------------------


def test_every_env_var_is_one_the_code_reads(config):
    # An env var nothing reads is a comment pretending to be configuration.
    sources = "".join(
        (REPO_ROOT / name).read_text(encoding="utf-8") for name in ("prepare.py", "train.py")
    )
    for key in config["env"]:
        assert key in sources, f"nothing reads {key}"


def test_artifacts_are_written_outside_the_worktree(config):
    # A one-epoch smoke test once overwrote a shipped 64-d embedding with a
    # 48-d model. train.py's default out dir is the working directory, which
    # in the container is the published worktree.
    assert config["env"]["AUTORESEARCH_OUT_DIR"] == "/out"


def test_read_only_paths_exist_and_are_directories(config):
    # herdmux skips a readOnly path that does not exist rather than failing,
    # so a stale or misspelled entry buys no protection and says nothing.
    assert config["readOnly"], "no readOnly paths declared"
    for rel in config["readOnly"]:
        assert (REPO_ROOT / rel).is_dir(), f"readOnly path {rel} is not a directory"


def test_committed_state_directories_are_read_only(config):
    # bb-offload holds committed queue state and programs/ the committed
    # experiment specs. Neither is an input a trainer has any reason to write.
    assert {"bb-offload", "programs"} <= set(config["readOnly"])


# --- container sizing ------------------------------------------------------


def test_shm_plus_memory_fits_the_docker_vm(config):
    # Over-commit and the container is OOM-killed mid-run, which reads as a
    # training failure rather than as a sizing mistake.
    total = _parse_size_gb(config["shmSize"]) + _parse_size_gb(config["memory"])
    assert total <= VM_MEMORY_GB, f"{total} GB requested, VM has {VM_MEMORY_GB} GB"


def test_timeout_is_a_positive_number_of_milliseconds(config):
    # On expiry herdmux kills the container and records `timeout` -- it never
    # reports an unfinished run as a pass.
    assert isinstance(config["timeoutMs"], int)
    assert config["timeoutMs"] > 0
