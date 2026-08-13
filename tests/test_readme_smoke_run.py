"""Check README's smoke-run section against the code it tells a stranger to run.

The section exists because the only other documented path is a 20-experiment
autonomous loop needing a GitHub token -- there was no way to run the trainer
once and see it work. A doc that walks someone through a run is only worth
having while it is true, and every claim in it is a claim about a specific line
of prepare.py or train.py. These tests pin the ones whose drift would cost the
reader something real: a wrong entrypoint, a checkpoint written into their
checkout, a quoted message that no longer exists, or a renamed output key the
section tells them to go read.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def readme():
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_names_both_entrypoints_in_order(readme):
    # train.py does not prepare data; a reader who skips prepare.py gets a
    # missing-parquet failure well after `uv sync` has been paid for. Matched on
    # the script names rather than the full `uv run ...` line, because the
    # container lane spells the same two steps `python -u prepare.py` -- the
    # order is the claim, not the launcher.
    assert "prepare.py" in readme
    assert "train.py --smoke-test" in readme
    assert readme.index("prepare.py") < readme.index("train.py --smoke-test")


def test_readme_tells_the_reader_to_set_the_out_dir(readme):
    # _resolve_out_dir defaults to the working directory and
    # _save_pre_eval_checkpoint runs on every path, --smoke-test included. A
    # reader who follows this section without AUTORESEARCH_OUT_DIR set drops
    # checkpoint_pre_eval.pt into their checkout -- the one way this section can
    # actively harm someone.
    assert "AUTORESEARCH_OUT_DIR" in readme
    assert "checkpoint_pre_eval.pt" in readme


@pytest.mark.parametrize(
    "message",
    (
        # detect_runtime's guard: the section promises there is no CPU path.
        "CUDA is required. No CUDA device detected.",
        # _compatibility_warning's laptop case: the section tells the reader
        # this one is expected output, not a fault to chase. Quoting a string
        # nothing prints would make that reassurance unverifiable.
        "laptop GPUs are outside the supported desktop matrix",
    ),
)
def test_readme_quotes_messages_train_py_still_emits(readme, message):
    train_py = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
    assert message in readme, f"README no longer quotes {message!r}"
    assert message in train_py, f"train.py no longer emits {message!r}"


def test_readme_names_the_four_loop_diagnostic_keys(readme):
    # The keys are the section's payload: they are what a reader greps the log
    # for, and loop_bound is the verdict the whole GPU lane exists to produce.
    # train.py prints them from _format_input_bound_lines; if that renaming ever
    # happens, the README's "read this" instruction points at nothing.
    train_py = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
    for key in ("dataloader_percent", "gpu_wait_percent", "other_percent", "loop_bound"):
        assert key in readme, f"README does not name {key}"
        assert f"{key}: " in train_py, f"train.py no longer prints a {key!r} line"
