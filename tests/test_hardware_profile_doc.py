"""Hold docs/HARDWARE_PROFILE.md's decision rule to the code that decides.

The doc's "Is the loop input-bound?" section tells a reader how to convert two
printed numbers into a decision: at what `dataloader_percent` to stop tuning the
GPU and go fix the input pipeline instead. Those cutoffs are not the doc's to
choose -- `train.py` owns them as module-level constants, and the doc is quoting
them. Change a constant and the doc keeps quoting the old one in prose that still
reads as authoritative, so a reader acts on a rule the trainer no longer applies.
That is the worst kind of stale documentation: confidently wrong and cheap to
believe.

Scope is deliberately narrow. This checks the *decision rule* -- the three
thresholds and the three verdict labels -- and nothing else. In particular it
asserts nothing about the recorded-runs table: those rows are a dated historical
record of what the lane printed, and pinning them to today's constants would
falsify history the moment a threshold moves.

String and AST checks on files. No GPU, no torch, no container.
"""

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "docs" / "HARDWARE_PROFILE.md"
TRAIN_PATH = REPO_ROOT / "train.py"

# utf-8-sig, not utf-8, wherever text is handed to a parser: a BOM survives a
# plain utf-8 decode as a leading U+FEFF, which makes ast.parse raise. It is a
# no-op on BOM-less files.
PARSE_ENCODING = "utf-8-sig"

# Each entry: the constant train.py owns, and the regex matching how the doc's
# "Reading it" bullets quote it. The doc states percents; the constants are
# fractions. Each pattern appears exactly once in the doc -- the surrounding
# backticks and comparison operator are part of the match, so the looser prose
# elsewhere ("an `input-bound` threshold of 50") cannot satisfy it by accident.
THRESHOLD_QUOTES = (
    ("INPUT_BOUND_DATA_FRACTION", r"`dataloader_percent >= (\d+)`"),
    ("INPUT_BOUND_GPU_WAIT_FRACTION", r"`gpu_wait_percent < (\d+)`"),
    ("COMPUTE_BOUND_GPU_WAIT_FRACTION", r"`gpu_wait_percent >= (\d+)`"),
)

CLASSIFIER = "_classify_step_bound"


@pytest.fixture(scope="module")
def doc():
    return DOC_PATH.read_text(encoding=PARSE_ENCODING)


@pytest.fixture(scope="module")
def train_tree():
    return ast.parse(TRAIN_PATH.read_text(encoding=PARSE_ENCODING))


def _module_constants(tree):
    """Module-scope `NAME = <literal>` assignments, as a dict.

    Module scope only (`tree.body`, not `ast.walk`): a same-named local inside
    some function is not what the doc is quoting.
    """
    constants = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Most module-level assignments in train.py are not literals
                # (`torch.device(...)`, `os.environ.get(...)`); skip them
                # rather than let one abort the scan.
                try:
                    constants[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    pass
    return constants


def _verdict_labels(tree):
    """The string literals `_classify_step_bound` assigns to `verdict`."""
    func = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == CLASSIFIER
        ),
        None,
    )
    assert func is not None, f"train.py defines no {CLASSIFIER}"
    labels = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if "verdict" in names and isinstance(node.value.value, str):
                labels.add(node.value.value)
    return labels


def test_the_doc_exists(doc):
    assert doc.strip(), "docs/HARDWARE_PROFILE.md is empty"


def test_the_doc_quotes_the_thresholds_train_py_actually_applies(doc, train_tree):
    # A reader who follows the doc's rule should reach the same verdict the
    # trainer printed. Move a constant without moving the prose and they do not.
    constants = _module_constants(train_tree)
    for name, pattern in THRESHOLD_QUOTES:
        assert name in constants, f"train.py no longer defines {name} at module scope"
        found = re.findall(pattern, doc)
        assert len(found) == 1, (
            f"expected exactly one quotation of {name} matching {pattern!r}, found {len(found)}"
        )
        assert int(found[0]) == round(constants[name] * 100), (
            f"doc says {found[0]}% for {name}, train.py uses {constants[name]}"
        )


def test_the_doc_names_every_verdict_the_classifier_can_print(doc, train_tree):
    # A verdict the doc does not explain is a run nobody can read. `mixed` is
    # the one that matters here: it is what this box actually prints, and it is
    # the label most easily mistaken for "the measurement failed".
    labels = _verdict_labels(train_tree)
    assert labels, f"{CLASSIFIER} assigns no string verdicts -- has it been rewritten?"
    for label in sorted(labels):
        assert label in doc, f"{CLASSIFIER} can print {label!r}, which the doc never explains"
