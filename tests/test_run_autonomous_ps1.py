"""Tests for scripts/run-autonomous.ps1 — the loop that produces every artifact
the rest of the repo reads.

This 190-line script had no tests. It writes ``results.tsv`` and
``bb-offload/results/results.jsonl``, and those two files are the sole input to
``bb rtx results --best``, ``bb rtx sync``, ``bb rtx status`` and the release
notes ``publish-release.ps1`` puts on the GitHub release page. Everything
downstream trusts it. Two defects lived in that gap. Both were reproduced by
these tests against the real, unfixed script before the fix was applied: with
the parse defect repaired but the loop otherwise untouched, the four tests below
failed and the two lock-in tests passed. A third defect — the script did not
parse at all, so none of this could run — is covered by
test_ps1_scripts_parse.py:

1. **A failed training run was logged as a good one.** The script decided
   success by looking for a ``val_bpb:`` line in ``run.log`` and never read the
   exit code of ``uv run train.py`` at all — ``$LASTEXITCODE`` appeared nowhere
   in the file. Absence of ``val_bpb:`` correctly meant crash, but *presence* of
   it did not mean success: ``train.py`` prints ``val_bpb:`` at line 1488 and
   then prints ~30 more lines before ``return 0``, so anything that kills the
   process after that first print — a ``KeyError`` in the reporting block (which
   the trainer agent edits every cycle), a CUDA teardown error at interpreter
   exit, a Ctrl-C — left a parsed, plausible ``val_bpb`` in the log and the
   script wrote ``status=keep`` for it. The try/catch around the run does not
   help: a native command exiting nonzero does not throw in PowerShell, so that
   catch only ever fires for command-not-found.

   This is worse than a mislabelled row because of *how* the consumers filter.
   ``results --best`` (cli.py:172) and ``sync`` (cli.py:204) select on
   ``val_bpb > 0`` and never look at ``status``. So a nonzero-exit run carrying a
   positive ``val_bpb`` is not merely recorded — it is eligible to win
   best-of-run and be published as the headline number. Hence the fix zeroes
   ``val_bpb`` rather than only correcting ``status``: the producer is the only
   layer that can keep this out, and ``val_bpb > 0`` is the repo's existing
   "this measurement is real" contract. The rejected value is preserved as text
   in the description column so a human still sees what the run printed.

2. **A run that never launched inherited the previous run's score.** ``run.log``
   was never cleared between iterations, and ``uv run train.py > run.log`` only
   truncates the file once the process actually starts. When ``uv`` could not be
   launched at all, PowerShell threw before the redirect ever touched the file,
   so the *previous* iteration's log was still on disk, ``Test-Path`` found it,
   and its ``val_bpb:`` line was parsed as this iteration's result — the same
   stale-state leak the comment at lines 48-53 already guards against for the
   in-memory variables.

``uv``, ``git`` and ``publish-release.ps1`` are stubbed in a sandbox repo so the
tests exercise the real control flow without a GPU or a network, and PATH is
reduced to the stub directory plus System32 so a real tool can never be resolved
by accident. The ``publish-release.ps1`` stub is deliberate: the final-publish
call at line 184 runs on every invocation and that script already has its own
eight tests (test_publish_release_ps1.py); stubbing it keeps a failure here
pointing at the loop rather than at the publisher.
"""

import functools
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

if os.name != "nt":
    # Not a masked environment defect: run-autonomous.ps1 is Windows-only
    # (backtick escapes, `Out-File -Encoding utf8`, .cmd stub resolution) and
    # has no POSIX twin, so there is nothing to assert on elsewhere.
    pytest.skip(
        "run-autonomous.ps1 is Windows-only and has no POSIX twin",
        allow_module_level=True,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run-autonomous.ps1"

HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription"

# Stands in for `uv run train.py`. Emits the four summary lines the script greps
# for, in the relative order train.py prints them (val_bpb, training_seconds,
# peak_vram_mb, num_steps), then exits with a caller-controlled code. The step
# count and the training time share one flag because they come from the same
# summary block: a run that printed one printed the other.
# Emission is gated on explicit 0/1
# flags rather than on empty strings: an unset variable in cmd expands to its
# own literal name (`%UV_VAL_BPB%`), which would put junk in the log instead of
# the intended nothing. stdout only -- the script redirects with `2>&1`, and
# PowerShell 5.1 wraps a native command's redirected stderr in a
# NativeCommandError, which would test PowerShell rather than this script.
UV_STUB = """@echo off
if "%UV_EXIT%"=="" set UV_EXIT=0
echo ---
if "%UV_EMIT_VAL%"=="1" echo val_bpb:          %UV_VAL_BPB%
if "%UV_EMIT_STEPS%"=="1" echo training_seconds: %UV_TRAIN_SEC%
if "%UV_EMIT_MEM%"=="1" echo peak_vram_mb:     %UV_PEAK_MB%
if "%UV_EMIT_STEPS%"=="1" echo num_steps:        %UV_NUM_STEPS%
exit /b %UV_EXIT%
"""

# `git branch --list` must print nothing so the script takes the checkout -b
# branch; `rev-parse --short HEAD` must print a hash for the commit column.
GIT_STUB = """@echo off
if "%1"=="branch" exit /b 0
if "%1"=="rev-parse" (
  echo abc1234
  exit /b 0
)
exit /b 0
"""

# The loop calls this at line 184 on every run. Covered by its own tests.
PUBLISH_STUB = """param(
    [string]$Program = "",
    [string]$Tag = ""
)
Write-Host "publish-release stub: $Tag"
exit 0
"""


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
            "Cannot execute scripts/run-autonomous.ps1: neither 'pwsh' nor "
            "'powershell' is on PATH. This is an environment failure, not a "
            "defect in run-autonomous.ps1.",
            pytrace=False,
        )
    return found


def _sandbox(tmp_path, uv_missing=False):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copy(SCRIPT, scripts_dir / "run-autonomous.ps1")
    # $PSScriptRoot resolves to this dir, so the loop finds the stub publisher.
    (scripts_dir / "publish-release.ps1").write_text(PUBLISH_STUB)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if not uv_missing:
        (bin_dir / "uv.cmd").write_text(UV_STUB)
    (bin_dir / "git.cmd").write_text(GIT_STUB)
    return scripts_dir / "run-autonomous.ps1", bin_dir


def _run(
    tmp_path,
    *args,
    uv_exit=0,
    val_bpb="0.981200",
    peak_mb="11264.0",
    num_steps="742",
    training_seconds="300.4",
    emit_val=True,
    emit_mem=True,
    emit_steps=True,
    uv_missing=False,
):
    """Run the real loop against the stubs. Defaults to one clean experiment."""
    script, bin_dir = _sandbox(tmp_path, uv_missing=uv_missing)

    # Hermetic PATH: only the stubs plus System32. Without this, `uv_missing`
    # would silently resolve a real uv on a machine that has one -- and this
    # test would launch an actual training run.
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(bin_dir), str(system32)])
    env["UV_EXIT"] = str(uv_exit)
    env["UV_VAL_BPB"] = val_bpb
    env["UV_PEAK_MB"] = peak_mb
    env["UV_NUM_STEPS"] = num_steps
    env["UV_TRAIN_SEC"] = training_seconds
    env["UV_EMIT_VAL"] = "1" if emit_val else "0"
    env["UV_EMIT_MEM"] = "1" if emit_mem else "0"
    env["UV_EMIT_STEPS"] = "1" if emit_steps else "0"

    return subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-MaxExperiments",
            "1",
            *args,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )


def _rows(tmp_path, proc=None):
    """Data rows of results.tsv, split into columns. Out-File writes a BOM."""
    path = tmp_path / "results.tsv"
    if not path.exists() and proc is not None:
        pytest.fail(
            "run-autonomous.ps1 produced no results.tsv at all. Script output:\n"
            + proc.stdout
            + proc.stderr,
            pytrace=False,
        )
    text = path.read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    assert lines[0] == HEADER, f"unexpected header: {lines[0]!r}"
    return [ln.split("\t") for ln in lines[1:]]


def _jsonl(tmp_path):
    path = tmp_path / "bb-offload" / "results" / "results.jsonl"
    text = path.read_text(encoding="utf-8-sig")
    return [json.loads(ln) for ln in text.strip().splitlines() if ln.strip()]


# --- defect 1: a nonzero exit must not be laundered into a good result ----


def test_nonzero_exit_after_val_bpb_is_not_logged_as_keep(tmp_path):
    """train.py printed a score and then died. That is not a keeper."""
    proc = _run(tmp_path, uv_exit=1, val_bpb="0.981200")

    (row,) = _rows(tmp_path, proc)
    _commit, logged_bpb, _mem, status, _desc = row[:5]
    assert status != "keep", (
        f"a run whose process exited 1 was recorded as a keeper; row={row!r}"
    )
    assert float(logged_bpb) == 0.0, (
        "the score from a failed run stayed positive, so `results --best` and "
        f"`sync` (which filter only on val_bpb > 0) can still promote it: {row!r}"
    )


def test_nonzero_exit_cannot_win_best_of_run(tmp_path):
    """The JSONL row is what cli.py reads; enforce the val_bpb > 0 contract there."""
    _run(tmp_path, uv_exit=1, val_bpb="0.500000")

    (rec,) = _jsonl(tmp_path)
    assert rec["status"] != "keep", rec
    # Mirrors the filters at cli.py:172 (results --best) and cli.py:204 (sync).
    assert not (isinstance(rec["val_bpb"], (int, float)) and rec["val_bpb"] > 0), (
        f"a failed run is still eligible to be published as the best result: {rec}"
    )


def test_rejected_run_records_the_exit_code_and_the_value_it_printed(tmp_path):
    """Demote the row, but do not silently destroy what the run reported."""
    proc = _run(tmp_path, uv_exit=137, val_bpb="0.777700")

    (row,) = _rows(tmp_path, proc)
    desc = row[4]
    assert "137" in desc, f"exit code not recoverable from the log row: {desc!r}"
    assert "0.7777" in desc, f"rejected score not preserved for a human: {desc!r}"
    assert "137" in proc.stdout, "the demotion was not reported on the console"


# --- defect 2: a run that never launched must not inherit a stale score ---


def test_failed_launch_does_not_reuse_the_previous_run_log(tmp_path):
    """`uv` cannot be launched at all, so the redirect never truncates run.log.

    The leftover log from an earlier experiment must not be read as this one's
    result. `uv_missing` removes the stub entirely, which is the only faithful
    way to reproduce this: a stub that runs and then fails still truncates the
    log on the way in, which quietly turns this into the already-working
    no-val_bpb crash path and the test would pass against the unfixed script.
    """
    (tmp_path / "run.log").write_text(
        "---\nval_bpb:          0.412300\npeak_vram_mb:     9000.0\n"
    )
    proc = _run(tmp_path, uv_missing=True)

    (row,) = _rows(tmp_path, proc)
    logged_bpb, status = row[1], row[3]
    assert float(logged_bpb) == 0.0, (
        f"stale run.log from a previous experiment was parsed as this one: {row!r}"
    )
    assert status != "keep", row


# --- the paths that already worked: lock them in -------------------------


def test_clean_run_is_logged_as_keep(tmp_path):
    """A fix that demoted everything would pass the tests above for free."""
    proc = _run(tmp_path, uv_exit=0, val_bpb="0.981200", peak_mb="11264.0")

    (row,) = _rows(tmp_path, proc)
    commit, logged_bpb, mem_gb, status, _desc = row[:5]
    assert status == "keep", f"a clean run was not kept: {row!r}"
    assert float(logged_bpb) == pytest.approx(0.9812)
    assert float(mem_gb) == pytest.approx(11.0)
    assert commit == "abc1234"

    (rec,) = _jsonl(tmp_path)
    assert rec["status"] == "keep"
    assert rec["val_bpb"] == pytest.approx(0.9812)


def test_missing_val_bpb_is_a_crash(tmp_path):
    """The detection that already worked: the run emitted no score line."""
    proc = _run(tmp_path, uv_exit=1, emit_val=False, emit_mem=False, emit_steps=False)

    (row,) = _rows(tmp_path, proc)
    assert float(row[1]) == 0.0
    assert row[3] == "crash", row


# --- what makes two val_bpb rows comparable --------------------------------
#
# A real run trains to a wall-clock budget (train.py:1338 breaks on
# `total_training_time >= TIME_BUDGET`, and `max_steps` is None off the smoke
# path), so the number of optimizer steps -- how much data the model saw -- is
# set by how fast the box happened to run: thermals, the autotuned batch size,
# anything else sharing the GPU. A lower val_bpb can therefore mean a better
# model or merely a faster machine-hour, and nothing in the recorded row told
# the two apart. train.py prints both numbers, but run.log is deleted at the top
# of the next iteration (line 69), so the row is the only place they survive.


def test_clean_run_records_what_makes_its_score_comparable(tmp_path):
    _run(tmp_path, uv_exit=0, num_steps="742", training_seconds="300.4")

    (rec,) = _jsonl(tmp_path)
    assert rec["num_steps"] == 742, (
        "the step count is absent from the recorded result, so a reader cannot "
        f"tell a better model from a faster machine-hour: {rec}"
    )
    assert rec["training_seconds"] == pytest.approx(300.4), rec


def test_failed_launch_does_not_inherit_the_previous_step_count(tmp_path):
    """The mirror of the stale-log defect, for the new fields.

    Parsed only in the success branch and left out of the per-iteration reset,
    these would carry the previous experiment's values into a row whose score is
    0 -- a crash wearing a completed run's step count.
    """
    (tmp_path / "run.log").write_text(
        "---\nval_bpb:          0.412300\ntraining_seconds: 299.9\n"
        "peak_vram_mb:     9000.0\nnum_steps:        900\n"
    )
    proc = _run(tmp_path, uv_missing=True)

    (rec,) = _jsonl(tmp_path)
    assert rec["num_steps"] == 0, (
        f"stale step count from a previous experiment leaked into this row: {rec}"
    )
    assert rec["training_seconds"] == 0.0, rec
    assert float(_rows(tmp_path, proc)[0][1]) == 0.0


def test_rejected_run_does_not_keep_a_completed_runs_step_count(tmp_path):
    """train.py printed a summary and then exited nonzero: the score is zeroed,
    and the step count describing it has to go with it."""
    _run(tmp_path, uv_exit=1, num_steps="742", training_seconds="300.4")

    (rec,) = _jsonl(tmp_path)
    assert rec["val_bpb"] == 0.0
    assert rec["num_steps"] == 0, (
        f"a rejected run still looks like a completed measurement: {rec}"
    )
    assert rec["training_seconds"] == 0.0, rec
