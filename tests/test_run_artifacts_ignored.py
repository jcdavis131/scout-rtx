"""Check that everything a run writes into the worktree is ignored.

The failure this guards is the one that cost this project the most: a run
started only to prove the trainer works also ships an artifact. train.py's
_save_pre_eval_checkpoint is unconditional -- --smoke-test included -- and
_resolve_out_dir defaults to the working directory, so the artifact lands in
the checkout unless AUTORESEARCH_OUT_DIR points elsewhere. Neither shipped
caller sets it: scripts/setup-win.ps1:52 runs `uv run train.py --smoke-test`
from the repo root unconditionally, and scripts/run-autonomous.ps1:76 runs
`uv run train.py` from the repo root once per experiment in its loop.
herdmux.train.json's env block is the only thing in the repo that sets the
variable, and it only covers the container lane. README tells a reader to set
it by hand; this test is the net for the paths where nobody does.

That net has to be a rule and not a filename. A second artifact added under
out_dir later would be just as unignored, and nothing else in the suite would
notice -- so these tests read the artifact names out of train.py rather than
hardcoding them, and fail if that read comes back empty.

Deliberately NOT covered here: the four `=0.1.0`-style files in the repo root,
junk from a `bash -lc` redirect of an unquoted pip specifier. Ignoring them
would make the next occurrence invisible instead of preventing it, and the
cause is already pinned at its source by
test_train_lane_config.test_requirements_are_quoted_against_shell_redirection.
They are committed, so only a `git rm` clears them. Nor .pytest_cache/: pytest
writes a `*` .gitignore inside it, so it already ignores itself.

No torch, no git -- this runs anywhere the suite does.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# `os.path.join(out_dir, "checkpoint_pre_eval.pt")` -> the basename a run drops
# into whatever _resolve_out_dir() returned.
OUT_DIR_ARTIFACT_RE = re.compile(r"""os\.path\.join\(\s*out_dir\s*,\s*["']([^"']+)["']""")


@pytest.fixture(scope="module")
def run_artifacts():
    """Basenames train.py writes under _resolve_out_dir()."""
    train_py = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
    return sorted(set(OUT_DIR_ARTIFACT_RE.findall(train_py)))


@pytest.fixture(scope="module")
def ignored_patterns():
    """Non-comment, non-blank lines of .gitignore."""
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    return [
        stripped
        for stripped in (line.strip() for line in text.splitlines())
        if stripped and not stripped.startswith("#")
    ]


def test_the_artifact_scan_actually_found_something(run_artifacts):
    # Guards the vacuous pass. If out_dir is ever renamed or the join is
    # rewritten as an f-string, the regex below returns [] and every other test
    # in this file goes green while covering nothing -- a check that silently
    # answers zero is worse than no check. Fail here instead, loudly, so the
    # next author updates the pattern rather than inheriting a dead test.
    assert run_artifacts, (
        "found no os.path.join(out_dir, ...) artifacts in train.py; "
        "OUT_DIR_ARTIFACT_RE no longer matches how the code writes them"
    )


def test_every_out_dir_artifact_is_gitignored(run_artifacts, ignored_patterns):
    # Matched as an exact line rather than by implementing gitignore glob
    # semantics: every entry that needs to satisfy this is a plain filename, and
    # a half-right glob matcher would be its own source of false confidence.
    unignored = [name for name in run_artifacts if name not in ignored_patterns]
    assert not unignored, (
        f"train.py writes {unignored} into AUTORESEARCH_OUT_DIR, which defaults "
        f"to the working directory, and .gitignore does not cover them"
    )


def test_the_checkpoint_is_still_written_unconditionally():
    # The reason the .gitignore entry is load-bearing rather than tidy. If this
    # ever becomes gated on a --write-artifacts-style flag, the entry stops
    # being a safety net and this test should be the thing that says so.
    train_py = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
    # >= 2, not `in`: the definition alone satisfies a substring check, so a
    # deleted call site would leave this green while the test's name claims the
    # opposite. Two occurrences means def plus at least one caller. Whether that
    # caller is *unconditional* is not something a static read can pin -- that
    # part lives in the call site itself and is why the name says so out loud.
    assert train_py.count("_save_pre_eval_checkpoint(") >= 2, (
        "_save_pre_eval_checkpoint is defined but never called; "
        "re-check whether a run still writes a checkpoint at all"
    )
    assert 'os.environ.get("AUTORESEARCH_OUT_DIR") or "."' in train_py, (
        "_resolve_out_dir no longer defaults to the working directory; "
        "re-check whether an unset AUTORESEARCH_OUT_DIR still writes into the checkout"
    )
