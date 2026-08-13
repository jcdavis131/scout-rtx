"""Tests for scripts/publish-release.sh: best-value awk + missing-results guard.

These tests execute the real shell script, so they need a bash that can actually
run a script file. Picking one with a bare ``subprocess.run(["bash", ...])`` lets
PATH order decide: on Windows the first ``bash`` is often the WSL launcher
(``C:\\Windows\\System32\\bash.exe``), which dies with
``execvpe(/bin/bash) failed: No such file or directory`` when no distro is
installed. That surfaced here as ``assert 1 == 0`` on a product assertion — a
missing interpreter wearing the clothes of a broken publish script.

``_resolve_bash`` instead probes *every* bash candidate with a miniature of the
real harness and returns the absolute path of the first one that works, so PATH
order can no longer decide whether these tests pass. If none works, the tests
fail loudly naming each candidate and why it was rejected, rather than skipping
(a skip is exit 0 with the script never exercised, which reads as green).
"""

import functools
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "publish-release.sh"

HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription"

# --- bash resolution -----------------------------------------------------

# A miniature of what publish-release.sh actually needs: `set -e`, command
# substitution, `dirname`, array append, printf redirection, ${arr[idx]}, awk,
# and -- the part that matters most -- a *stubbed* `git` that must win over the
# real git on PATH.
#
# Probing with a name that has no real counterpart (`gh`) is not enough:
# C:\\Program Files\\Git\\bin\\bash.exe rewrites PATH so /mingw64/bin/git shadows
# an injected stub. The real script then runs `git rev-parse` in a non-repo temp
# dir, gets exit 128 with stderr swallowed by 2>/dev/null, and `set -e` kills it
# -- an empty-output failure that looks nothing like a PATH problem. So the probe
# stubs `git` too and requires the stub's marker back.
_PROBE_SCRIPT = """#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
ITEMS=()
ITEMS+=("probe-ok")
printf 'header\\n' > probe.txt
gh marker >> probe.txt
awk 'NR==2{print $1}' probe.txt
echo "rev=$(git rev-parse --short HEAD 2>/dev/null)"
echo "${ITEMS[0]}"
"""

# Every marker must come back, or this bash cannot run the real harness.
_PROBE_MARKERS = ("gh-stub", "probe-git-stub", "probe-ok")


def _bash_candidates():
    """Every plausible bash, in preference order, de-duplicated."""
    seen = set()
    found = []

    def add(path):
        path = os.path.normpath(str(path))
        key = path.lower() if os.name == "nt" else path
        if key not in seen and os.path.isfile(path):
            seen.add(key)
            found.append(path)

    names = ("bash.exe", "bash") if os.name == "nt" else ("bash",)
    # Every bash on PATH in PATH order -- not just the first, because the first
    # is frequently the WSL launcher.
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry:
            for name in names:
                add(Path(entry) / name)

    # Git for Windows ships a working bash that is usually not first on PATH.
    if os.name == "nt":
        for var in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
            base = os.environ.get(var)
            if base:
                add(Path(base) / "Git" / "usr" / "bin" / "bash.exe")
                add(Path(base) / "Git" / "bin" / "bash.exe")
    return found


def _harness_env(bash_path, stub_dir=None):
    """Child PATH: stubs first, then the chosen bash's own directory.

    The stubs carry `#!/usr/bin/env bash` shebangs, so without that second entry
    the child resolves `bash` through the ambient PATH all over again and can
    land on a different (broken) bash than the one we picked -- the same
    PATH-order coin flip this module exists to remove, one level down.
    """
    env = dict(os.environ)
    parts = [] if stub_dir is None else [str(stub_dir)]
    parts.append(os.path.dirname(bash_path))
    env["PATH"] = os.pathsep.join(parts + [env.get("PATH", "")])
    return env


def _probe_bash(candidate, workdir):
    """Run the miniature harness. Returns None if it works, else the reason."""
    bin_dir = workdir / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text('#!/usr/bin/env bash\necho "gh-stub $*"\nexit 0\n')
    git = bin_dir / "git"
    git.write_text("#!/usr/bin/env bash\necho probe-git-stub\nexit 0\n")
    for stub in (gh, git):
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    script = workdir / "probe.sh"
    script.write_text(_PROBE_SCRIPT)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)

    try:
        proc = subprocess.run(
            [candidate, str(script)],
            cwd=workdir,
            capture_output=True,
            text=True,
            env=_harness_env(candidate, bin_dir),
            timeout=60,
        )
    except OSError as exc:
        return "could not execute: {}".format(exc)
    except subprocess.TimeoutExpired:
        return "timed out after 60s"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().replace("\n", " ")
        return "exit {}: {}".format(proc.returncode, detail[:200])
    missing = [m for m in _PROBE_MARKERS if m not in proc.stdout]
    if missing:
        return "missing {} in probe output: {!r}".format(
            ", ".join(missing), proc.stdout.strip()[:200]
        )
    return None


@functools.lru_cache(maxsize=1)
def _resolve_bash():
    """(path, None) for the first working bash, else (None, diagnostic)."""
    candidates = _bash_candidates()
    if not candidates:
        return None, "no bash executable found on PATH or in the usual Git for Windows locations"
    rejected = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, candidate in enumerate(candidates):
            workdir = Path(tmp) / "probe{}".format(index)
            workdir.mkdir()
            reason = _probe_bash(candidate, workdir)
            if reason is None:
                return candidate, None
            rejected.append("  {}\n      {}".format(candidate, reason))
    return None, "no working bash found; tried:\n" + "\n".join(rejected)


def _bash():
    path, reason = _resolve_bash()
    if path is None:
        pytest.fail(
            "Cannot execute scripts/publish-release.sh: no working bash.\n"
            + reason
            + "\nThis is an environment failure, not a defect in "
            "publish-release.sh. Install Git for Windows (or any POSIX bash) "
            "to run these tests.",
            pytrace=False,
        )
    return path


# --- the awk best-value expression ---------------------------------------


def _extract_best_line():
    """Pull the real BEST=$(awk ...) line out of the script so the test exercises
    the exact code that ships."""
    text = SCRIPT.read_text()
    match = re.search(r"^BEST=\$\(awk .*$", text, re.M)
    assert match, "BEST awk line not found in publish-release.sh"
    return match.group(0)


def _run_best(tmp_path):
    cmd = _extract_best_line() + '\necho "$BEST"'
    bash = _bash()
    proc = subprocess.run(
        [bash, "-c", cmd],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=_harness_env(bash),
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_awk_selects_minimum_val_bpb(tmp_path):
    rows = [
        "c1\t1.0500\t10.1\tkeep\tfirst",
        "c2\t0.9812\t11.0\tkeep\tbest",
        "c3\t1.1000\t12.2\tkeep\tworst",
    ]
    (tmp_path / "results.tsv").write_text("\n".join([HEADER] + rows) + "\n")
    assert float(_run_best(tmp_path)) == 0.9812


def test_awk_min_when_best_row_is_first(tmp_path):
    rows = ["c1\t0.9000\t10\tkeep\ta", "c2\t0.9500\t10\tkeep\tb"]
    (tmp_path / "results.tsv").write_text("\n".join([HEADER] + rows) + "\n")
    assert float(_run_best(tmp_path)) == 0.9


def test_awk_missing_file_reports_unknown(tmp_path):
    assert _run_best(tmp_path) == "unknown"


# --- the script end to end -----------------------------------------------


def _stub_bin(tmp_path):
    """Fake gh/git so the full script can run without network or a git repo."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text("#!/usr/bin/env bash\necho \"gh-stub $*\"\nexit 0\n")
    git = bin_dir / "git"
    git.write_text("#!/usr/bin/env bash\necho abc1234\nexit 0\n")
    for f in (gh, git):
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _script_copy(tmp_path):
    """Copy the script into a temp repo layout so its `cd ROOT` lands in tmp."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    dst = scripts_dir / "publish-release.sh"
    shutil.copy(SCRIPT, dst)
    dst.chmod(dst.stat().st_mode | stat.S_IEXEC)
    return dst


def _run_script(tmp_path, *args, env_path):
    bash = _bash()
    return subprocess.run(
        [bash, str(_script_copy(tmp_path)), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=_harness_env(bash, env_path),
    )


def test_script_errors_without_results_tsv(tmp_path):
    bin_dir = _stub_bin(tmp_path)
    proc = _run_script(tmp_path, "programs/program-ava.md", "v0.6.0-test", env_path=bin_dir)
    assert proc.returncode == 1
    assert "no results.tsv" in proc.stderr
    assert not (tmp_path / "results.tsv").exists()


def test_script_demo_flag_creates_demo_tagged_row(tmp_path):
    bin_dir = _stub_bin(tmp_path)
    proc = _run_script(tmp_path, "programs/program-ava.md", "v0.6.0-test", "--demo", env_path=bin_dir)
    assert proc.returncode == 0, proc.stderr
    tsv = (tmp_path / "results.tsv").read_text().strip().splitlines()
    assert tsv[0] == HEADER
    assert "\tdemo\t" in tsv[1]  # status column is 'demo', not 'keep'


def test_script_real_results_publishes_min(tmp_path):
    bin_dir = _stub_bin(tmp_path)
    rows = ["c1\t1.0500\t10\tkeep\ta", "c2\t0.9812\t11\tkeep\tb"]
    (tmp_path / "results.tsv").write_text("\n".join([HEADER] + rows) + "\n")
    proc = _run_script(tmp_path, "programs/program-ava.md", "v0.6.0-test", env_path=bin_dir)
    assert proc.returncode == 0, proc.stderr
    assert "Best: 0.9812" in proc.stdout
