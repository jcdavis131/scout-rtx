"""Tests for scripts/publish-release.ps1 — the publish path the README documents.

The POSIX twin (``publish-release.sh``) has had tests since an earlier cycle, but
the PowerShell script is the one the Quickstart tells a stranger to run and the
one ``run-autonomous.ps1`` invokes every five experiments, and it had none. Two
failure modes lived in that gap, both reproduced against the real script before
these tests were written:

1. **A failed publish reported success.** ``$ErrorActionPreference = "Stop"`` does
   *not* make a native command's nonzero exit terminate the script — only cmdlet
   errors do. So when ``gh release create``/``upload``/``edit`` failed, execution
   fell straight through to ``Write-Host "Done. Dashboard will auto-read in
   <60s..."`` and the script exited 0. The caller in ``run-autonomous.ps1`` wraps
   the call in try/catch, so its "Auto-publish failed" warning could never fire.

2. **The normal first publish could not run at all.** Windows PowerShell 5.1 turns
   a native command's stderr into a ``NativeCommandError`` whenever that stream is
   redirected, and under ``Stop`` that error is terminating. ``gh release view``
   writes "release not found" to stderr for a tag that does not exist yet, so
   ``$exists = gh release view $Tag --json tagName 2>$null`` killed the script at
   the existence check — before it could take the create branch. Since
   ``run-autonomous.ps1`` mints a fresh ``-MMdd-HHmm`` tag for every publish, the
   release never exists, so that was the only branch it ever took. The same
   redirect on ``git rev-parse ... 2>$null`` killed the "fall back to unknown"
   path outside a git repo, which is the exact defect the sibling ``.sh`` tests
   found on line 38 of the bash script.

``gh`` and ``git`` are stubbed as ``.cmd`` files on a PATH-prepended directory
(directory order beats PATHEXT, so these shadow the real executables) and record
what they were called with, so the tests assert the release was *actually*
created rather than just that the exit code looked happy. The ``gh`` stub fails
*selectively* — ``auth status`` always succeeds — because a stub that failed on
every call would make the create-failure test pass against the unfixed script for
the wrong reason: the script would die at the ``gh auth status`` guard on line 23
and never reach the code under test.
"""

import functools
import os
import shutil
import subprocess
from pathlib import Path

import pytest

if os.name != "nt":
    # Not a masked environment defect: this script is Windows-only (`Out-File
    # -Encoding utf8`, backtick escapes, .cmd stub resolution), so there is no
    # PowerShell-on-Linux run of it worth asserting on. The POSIX publish path
    # is covered by test_publish_release_sh.py, which does run everywhere.
    pytest.skip(
        "publish-release.ps1 is Windows-only; POSIX path covered by "
        "test_publish_release_sh.py",
        allow_module_level=True,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "publish-release.ps1"

HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription"
DONE_BANNER = "Done. Dashboard will auto-read"

# `gh release view` for a tag that does not exist yet: stderr + exit 1, exactly
# what the real gh does. `auth status` must keep succeeding regardless of
# GH_MUTATE_EXIT, or the script dies at its login guard instead of where we look.
GH_STUB = """@echo off
if "%GH_VIEW_EXIT%"=="" set GH_VIEW_EXIT=1
if "%GH_MUTATE_EXIT%"=="" set GH_MUTATE_EXIT=0
echo argv: %1 %2 %3 >> "%GH_LOG%"
echo notes: %7 >> "%GH_LOG%"
if "%1"=="auth" exit /b 0
if "%1 %2"=="release view" (
  if "%GH_VIEW_EXIT%"=="0" exit /b 0
  echo release not found 1>&2
  exit /b 1
)
if not "%GH_MUTATE_EXIT%"=="0" (
  echo gh: HTTP 422 upstream refused the upload 1>&2
  exit /b %GH_MUTATE_EXIT%
)
exit /b 0
"""

GIT_STUB = """@echo off
if "%GIT_NOT_A_REPO%"=="1" (
  echo fatal: not a git repository ^(or any of the parent directories^): .git 1>&2
  exit /b 128
)
echo abc1234
exit /b 0
"""


# --- powershell resolution ------------------------------------------------


@functools.lru_cache(maxsize=1)
def _resolve_powershell():
    for name in ("pwsh", "powershell"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _powershell():
    found = _resolve_powershell()
    if found is None:
        # Fail rather than skip: on Windows a missing PowerShell is a broken
        # environment, and a skip is exit 0 with the script never exercised.
        pytest.fail(
            "Cannot execute scripts/publish-release.ps1: neither 'pwsh' nor "
            "'powershell' is on PATH. This is an environment failure, not a "
            "defect in publish-release.ps1.",
            pytrace=False,
        )
    return found


# --- harness --------------------------------------------------------------


def _sandbox(tmp_path):
    """A temp repo layout so the script's `cd <parent of scripts/>` lands in tmp."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copy(SCRIPT, scripts_dir / "publish-release.ps1")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh.cmd").write_text(GH_STUB)
    (bin_dir / "git.cmd").write_text(GIT_STUB)
    return scripts_dir / "publish-release.ps1", bin_dir


def _run(tmp_path, *args, view_exit=1, mutate_exit=0, not_a_repo=False):
    """Run the real script against the stubs; return (proc, gh call log)."""
    script, bin_dir = _sandbox(tmp_path)
    gh_log = tmp_path / "gh-calls.log"

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")])
    env["GH_LOG"] = str(gh_log)
    env["GH_VIEW_EXIT"] = str(view_exit)
    env["GH_MUTATE_EXIT"] = str(mutate_exit)
    env["GIT_NOT_A_REPO"] = "1" if not_a_repo else "0"

    proc = subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            *args,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    log = gh_log.read_text() if gh_log.exists() else ""
    return proc, log


def _write_results(tmp_path, *rows):
    (tmp_path / "results.tsv").write_text("\n".join([HEADER, *rows]) + "\n")


def _read_results(tmp_path):
    # Out-File -Encoding utf8 writes a BOM in Windows PowerShell 5.1.
    return (tmp_path / "results.tsv").read_text(encoding="utf-8-sig")


BASE_ARGS = ("-Program", "programs/program-ava.md", "-Tag", "v0.6.0-test")


# --- the guards that already worked (lock them in) ------------------------


def test_errors_without_results_tsv(tmp_path):
    proc, _ = _run(tmp_path, *BASE_ARGS)
    assert proc.returncode != 0
    assert "no results.tsv" in (proc.stderr + proc.stdout)
    assert not (tmp_path / "results.tsv").exists(), "must not fabricate rows"


def test_demo_flag_seeds_demo_tagged_row(tmp_path):
    proc, _ = _run(tmp_path, *BASE_ARGS, "-Demo")
    assert proc.returncode == 0, proc.stderr
    lines = _read_results(tmp_path).strip().splitlines()
    assert lines[0] == HEADER
    assert "\tdemo\t" in lines[1], "synthetic row must be tagged status=demo"


def test_reports_minimum_val_bpb_as_best(tmp_path):
    _write_results(tmp_path, "c1\t1.0500\t10\tkeep\ta", "c2\t0.9812\t11\tkeep\tb")
    proc, _ = _run(tmp_path, *BASE_ARGS)
    assert proc.returncode == 0, proc.stderr
    assert "Best: 0.9812" in proc.stdout


def test_dry_run_publishes_nothing(tmp_path):
    _write_results(tmp_path, "c1\t0.9812\t11\tkeep\tb")
    proc, log = _run(tmp_path, *BASE_ARGS, "-DryRun")
    assert proc.returncode == 0, proc.stderr
    assert "release create" not in log and "release upload" not in log


# --- the two failures that reported success -------------------------------


def test_first_publish_creates_release_when_tag_absent(tmp_path):
    """The only branch run-autonomous.ps1 ever takes: the tag is always new.

    `gh release view` writes to stderr and exits 1 here, as the real gh does for
    an absent tag. Redirecting that stream under `Stop` used to kill the script
    at the existence check, so the release was never created.
    """
    _write_results(tmp_path, "c1\t0.9812\t11\tkeep\tb")
    proc, log = _run(tmp_path, *BASE_ARGS, view_exit=1)
    assert proc.returncode == 0, proc.stderr
    assert "release create" in log, "absent tag must take the create branch"


@pytest.mark.parametrize(
    "view_exit,failing_call",
    [(1, "release create"), (0, "release upload")],
    ids=["create-fails", "upload-fails"],
)
def test_failed_gh_call_is_not_reported_as_success(tmp_path, view_exit, failing_call):
    """A publish that published nothing must not exit 0 saying "Done.".

    Nonzero exits from a native command do not trip `$ErrorActionPreference =
    "Stop"`, so every gh failure used to fall through to the success banner.
    """
    _write_results(tmp_path, "c1\t0.9812\t11\tkeep\tb")
    proc, log = _run(tmp_path, *BASE_ARGS, view_exit=view_exit, mutate_exit=1)
    assert failing_call in log, "test must actually reach the call it checks"
    assert proc.returncode != 0, "failed publish exited 0"
    assert DONE_BANNER not in proc.stdout, "failed publish printed the success banner"


def test_commit_falls_back_to_unknown_outside_git_repo(tmp_path):
    """`git rev-parse ... 2>$null` intended a fallback; the redirect made it fatal."""
    _write_results(tmp_path, "c1\t0.9812\t11\tkeep\tb")
    proc, log = _run(tmp_path, "-Program", "programs/program-ava.md", "-Tag",
                     "v0.6.0-test", not_a_repo=True)
    assert proc.returncode == 0, proc.stderr
    assert "release create" in log
    assert "commit unknown" in log, "notes should record an unknown commit, not die"
