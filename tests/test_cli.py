"""Tests for bigbang-bridge/cli.py via typer's CliRunner with emit stubbed."""

import json

import httpx
import pytest
from typer.testing import CliRunner

runner = CliRunner()


# --- CUSTOM_ROOT resolution ------------------------------------------------

def test_custom_root_env_override(cli_mod, tmp_path):
    assert cli_mod.CUSTOM_ROOT == tmp_path / "rtx-root"


def test_custom_root_fallback_order(cli_mod, monkeypatch, tmp_path):
    monkeypatch.delenv("SCOUT_RTX_ROOT", raising=False)
    fake_home = tmp_path / "home"
    monkeypatch.setattr(cli_mod.Path, "home", classmethod(lambda cls: fake_home))
    # neither checkout exists -> scout-rtx fallback
    assert cli_mod._resolve_custom_root() == fake_home / "workspace" / "scout-rtx"
    # legacy checkout exists -> preferred
    legacy = fake_home / "workspace" / "autoresearch-rtx-custom"
    legacy.mkdir(parents=True)
    assert cli_mod._resolve_custom_root() == legacy


# --- queue add / list / clear ----------------------------------------------

def test_queue_add_list_clear(cli_mod, emit_records):
    result = runner.invoke(cli_mod.app, ["queue", "add", "--task", "tune router entropy", "--program", "programs/program-ava.md"])
    assert result.exit_code == 0
    added = emit_records[-1]
    assert added["added"]["task"] == "tune router entropy"
    assert added["added"]["status"] == "pending"
    assert cli_mod.QUEUE_FILE.exists()

    result = runner.invoke(cli_mod.app, ["queue", "list"])
    assert result.exit_code == 0
    listed = emit_records[-1]
    assert len(listed["tasks"]) == 1
    assert listed["tasks"][0]["program"] == "programs/program-ava.md"

    result = runner.invoke(cli_mod.app, ["queue", "clear"])
    assert result.exit_code == 0
    assert emit_records[-1]["cleared"] is True
    assert json.loads(cli_mod.QUEUE_FILE.read_text()) == {"tasks": []}


def test_queue_add_without_task_errors(cli_mod, emit_records):
    result = runner.invoke(cli_mod.app, ["queue", "add"])
    assert result.exit_code == 1
    assert "error" in emit_records[-1]


def test_queue_unknown_action(cli_mod, emit_records):
    result = runner.invoke(cli_mod.app, ["queue", "bogus"])
    assert result.exit_code == 0
    assert emit_records[-1]["valid"] == ["add", "list", "clear"]


# --- a corrupt queue is not an empty queue ---------------------------------

# The queue is hand-copied between the cloud session and the GPU box (it is the
# `next_steps` line `queue add` prints), so arriving truncated is a real failure
# mode. _load_queue used to answer `{"tasks": []}` for a file it could not
# parse, which made a corrupt queue indistinguishable from an empty one -- and
# the next `queue add` wrote that empty queue back over the pending tasks and
# reported the add as a success.

# A truncated copy of the shape bb-offload/queue.json actually has.
CORRUPT_QUEUE = '{"tasks": [{"id": "2026-07-15T21:23:49", "task": "tune router en'


def _seed_corrupt_queue(cli_mod, text=CORRUPT_QUEUE):
    cli_mod.QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    cli_mod.QUEUE_FILE.write_text(text, encoding="utf-8")
    return cli_mod.QUEUE_FILE.read_bytes()


def test_queue_add_refuses_to_overwrite_a_corrupt_queue(cli_mod, emit_records):
    """The one that matters: the bytes on disk must survive a refused add."""
    before = _seed_corrupt_queue(cli_mod)

    result = runner.invoke(cli_mod.app, ["queue", "add", "--task", "new work"])

    assert result.exit_code == 1
    assert cli_mod.QUEUE_FILE.read_bytes() == before, "add clobbered a queue it could not read"
    payload = emit_records[-1]
    assert "not valid JSON" in payload["error"]
    assert "queue clear" in payload["fix"], "a refusal with no way out is a dead end"


def test_queue_list_reports_a_corrupt_queue_instead_of_no_tasks(cli_mod, emit_records):
    _seed_corrupt_queue(cli_mod)

    result = runner.invoke(cli_mod.app, ["queue", "list"])

    assert result.exit_code == 1
    payload = emit_records[-1]
    assert "error" in payload
    assert payload.get("tasks") is None, "an unreadable queue reported as zero tasks"


def test_queue_clear_may_overwrite_but_says_so(cli_mod, emit_records):
    """clear is the escape hatch -- it discards by design, so it proceeds."""
    _seed_corrupt_queue(cli_mod)

    result = runner.invoke(cli_mod.app, ["queue", "clear"])

    assert result.exit_code == 0
    assert emit_records[-1]["cleared"] is True
    assert "not valid JSON" in emit_records[-1]["overwrote_unreadable"]
    assert json.loads(cli_mod.QUEUE_FILE.read_text()) == {"tasks": []}
    # and the reset queue is usable again
    assert runner.invoke(cli_mod.app, ["queue", "add", "--task", "x"]).exit_code == 0


def test_queue_add_refuses_valid_json_of_the_wrong_shape(cli_mod, emit_records):
    """A hand-edited `[]` used to die in an AttributeError traceback."""
    before = _seed_corrupt_queue(cli_mod, "[]")

    result = runner.invoke(cli_mod.app, ["queue", "add", "--task", "new work"])

    assert result.exit_code == 1
    assert cli_mod.QUEUE_FILE.read_bytes() == before
    assert "not a task queue" in emit_records[-1]["error"]


def test_status_reports_a_corrupt_queue_as_null_not_zero(cli_mod, emit_records):
    _seed_corrupt_queue(cli_mod)

    result = runner.invoke(cli_mod.app, ["status"])

    assert result.exit_code == 0, "status aggregates several sources; one bad file is not fatal"
    payload = emit_records[-1]
    assert payload["queue_pending"] is None, "an unreadable queue counted as 0 pending"
    assert payload["queue_total"] is None
    assert "not valid JSON" in payload["queue_error"]


def test_status_surfaces_an_unreadable_results_tsv(cli_mod, emit_records, monkeypatch):
    """An unreadable results.tsv read the same as a box that never ran anything."""
    _seed_tsv(cli_mod, "c1\t1.0500\t10\tkeep\ta")
    real_read_text = cli_mod.Path.read_text

    def explode(self, *args, **kwargs):
        if self == cli_mod.RESULTS_TSV:
            raise OSError("Input/output error")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(cli_mod.Path, "read_text", explode)

    result = runner.invoke(cli_mod.app, ["status"])

    assert result.exit_code == 0
    payload = emit_records[-1]
    assert payload["best"] == {}
    assert "Input/output error" in payload["best_error"]


def test_status_without_a_corrupt_queue_reports_no_error(cli_mod, emit_records):
    """The error keys are always present, so absence of a problem is explicit."""
    runner.invoke(cli_mod.app, ["queue", "add", "--task", "real work"])

    result = runner.invoke(cli_mod.app, ["status"])

    assert result.exit_code == 0
    payload = emit_records[-1]
    assert payload["queue_error"] is None
    assert payload["best_error"] is None
    assert payload["queue_pending"] == 1


# --- results TSV fallback returns LAST n rows ------------------------------

def test_results_tsv_fallback_returns_last_n(cli_mod, emit_records):
    header = "commit\tval_bpb\tmemory_gb\tstatus\tdescription"
    rows = [f"c{i}\t1.{i:02d}\t10\tkeep\trow{i}" for i in range(10)]
    cli_mod.RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    cli_mod.RESULTS_TSV.write_text("\n".join([header] + rows) + "\n")

    result = runner.invoke(cli_mod.app, ["results", "--n", "3"])
    assert result.exit_code == 0
    payload = emit_records[-1]
    assert payload["source"] == "results.tsv"
    assert payload["lines"][0] == header
    assert payload["lines"][1:] == rows[-3:]  # last 3, not first 3
    assert payload["count"] == 3


# --- status: a crashed run is not the best result --------------------------

# What run-autonomous.ps1 appends when train.py produced no `val_bpb:` line:
# val_bpb 0.0, status crash. Lower bpb is better, so 0 is the best value the
# column can hold -- an unfiltered minimum turns a crash into a record result.
CRASH_ROW = "c9\t0\t0\tcrash\tcrash"
NO_LOG_ROW = "c8\t0\t0\tcrash\tno log"


def _seed_tsv(cli_mod, *rows):
    header = "commit\tval_bpb\tmemory_gb\tstatus\tdescription"
    cli_mod.RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    cli_mod.RESULTS_TSV.write_text("\n".join([header, *rows]) + "\n")


def test_status_best_ignores_crashed_runs(cli_mod, emit_records):
    _seed_tsv(cli_mod, "c1\t1.0500\t10\tkeep\ta", CRASH_ROW, "c2\t0.9812\t11\tkeep\tb")

    result = runner.invoke(cli_mod.app, ["status"])
    assert result.exit_code == 0
    best = emit_records[-1]["best"]
    assert best["best_val_bpb"] == 0.9812, "a crash outranked every real run"
    assert best["best_commit"] == "c2"


def test_status_all_crash_reports_no_best(cli_mod, emit_records):
    """Nothing measured must read as nothing measured, not as a perfect score."""
    _seed_tsv(cli_mod, CRASH_ROW, NO_LOG_ROW)

    result = runner.invoke(cli_mod.app, ["status"])
    assert result.exit_code == 0
    assert emit_records[-1]["best"] == {}


# --- sync actually writes the MRR record (no longer a stub) ----------------

def test_sync_appends_mrr_record(cli_mod, emit_records):
    cli_mod.RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        {"ts": "t1", "commit": "aaa", "val_bpb": 1.02, "program": "program-ava.md"},
        {"ts": "t2", "commit": "bbb", "val_bpb": 0.98, "program": "program-ava.md"},
        {"ts": "t3", "commit": "ccc", "val_bpb": 1.10, "program": "program-base.md"},
    ]
    cli_mod.RESULTS_FILE.write_text("\n".join(json.dumps(l) for l in lines) + "\n")

    result = runner.invoke(cli_mod.app, ["sync"])
    assert result.exit_code == 0
    payload = emit_records[-1]
    assert payload["synced"] is True
    assert payload["best"]["commit"] == "bbb"

    written = [json.loads(l) for l in cli_mod.MRR_FILE.read_text().strip().splitlines()]
    assert len(written) == 1
    record = written[0]
    assert record["best"] == 0.98
    assert record["program"] == "program-ava.md"
    assert "ts" in record and "note" in record


def test_sync_without_results_is_honest(cli_mod, emit_records):
    result = runner.invoke(cli_mod.app, ["sync"])
    assert result.exit_code == 0
    assert emit_records[-1]["synced"] is False
    assert not cli_mod.MRR_FILE.exists()


# --- releases subcommand offline behavior ----------------------------------

def _raise_connect_error(*args, **kwargs):
    raise httpx.ConnectError("network unreachable")


def test_releases_list_offline_is_honest_nonzero(cli_mod, emit_records, monkeypatch):
    monkeypatch.setattr(cli_mod.httpx, "get", _raise_connect_error)
    result = runner.invoke(cli_mod.app, ["releases", "list"])
    assert result.exit_code == 1
    payload = emit_records[-1]
    assert payload["offline"] is True
    assert "api.github.com" in payload["error"]


def test_releases_sync_offline_is_honest_nonzero(cli_mod, emit_records, monkeypatch):
    monkeypatch.setattr(cli_mod.httpx, "get", _raise_connect_error)
    result = runner.invoke(cli_mod.app, ["releases", "sync", "--tag", "v0.6.0-ava-0716"])
    assert result.exit_code == 1
    assert emit_records[-1]["offline"] is True


def test_releases_list_parses_api_payload(cli_mod, emit_records, monkeypatch):
    fake_payload = [
        {
            "tag_name": "v0.6.0-ava-0716",
            "name": "v0.6.0-ava-0716 best 0.98",
            "published_at": "2026-07-16T00:00:00Z",
            "html_url": "https://github.com/jcdavis131/scout-rtx/releases/tag/v0.6.0-ava-0716",
            "assets": [{"name": "results.tsv", "size": 337, "browser_download_url": "https://example.invalid/results.tsv"}],
        }
    ]

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return fake_payload

    monkeypatch.setattr(cli_mod.httpx, "get", lambda *a, **k: FakeResponse())
    result = runner.invoke(cli_mod.app, ["releases", "list"])
    assert result.exit_code == 0
    payload = emit_records[-1]
    assert payload["count"] == 1
    assert payload["releases"][0]["tag"] == "v0.6.0-ava-0716"
    assert payload["releases"][0]["assets"][0]["name"] == "results.tsv"


def test_releases_sync_404_reports_missing_tag(cli_mod, emit_records, monkeypatch):
    request = httpx.Request("GET", "https://api.github.com/x")
    response = httpx.Response(404, request=request)

    def raise_404(*args, **kwargs):
        raise httpx.HTTPStatusError("not found", request=request, response=response)

    monkeypatch.setattr(cli_mod.httpx, "get", raise_404)
    result = runner.invoke(cli_mod.app, ["releases", "sync", "--tag", "v0.0.0-nope"])
    assert result.exit_code == 1
    assert "not found" in emit_records[-1]["error"]
