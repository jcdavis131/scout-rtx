"""Unit tests for pure-CPU pieces of train.py."""

import math

import pytest
import torch

import prepare
import train


# --- _resolve_gpu_profile tier boundaries ---------------------------------

ADA = (8, 9)
TURING = (7, 5)


def resolve(name="NVIDIA GeForce RTX 4080", cc=ADA, vram=16.0, is_windows=True):
    return train._resolve_gpu_profile(name, cc, vram, is_windows)


def test_16gb_card_underreporting_lands_in_ada_16gb():
    profile = resolve(vram=15.99)
    assert profile.name == "ada-16gb"
    assert profile.train_batch_candidates == (32, 16, 8, 4)
    assert profile.default_checkpointing is False


def test_24gb_card_underreporting_lands_in_ada_24gb_plus():
    profile = resolve(name="NVIDIA GeForce RTX 4090", vram=23.99)
    assert profile.name == "ada-24gb-plus"
    assert profile.train_batch_candidates == (64, 32, 16, 8, 4)


def test_exact_boundaries_keep_documented_tier():
    assert resolve(vram=16.0).name == "ada-16gb"
    assert resolve(name="RTX 4090", vram=24.0).name == "ada-24gb-plus"


def test_mid_tier_ada_12gb():
    profile = resolve(name="NVIDIA GeForce RTX 4070", vram=11.99)
    assert profile.name == "ada-10-15gb"
    assert profile.train_batch_candidates == (16, 8, 4)
    assert profile.default_checkpointing is True
    assert profile.eval_batch_cap == 16  # profile default


def test_below_16gb_tolerance_band_stays_mid_tier():
    # 15.4 is below the 15.5 tolerance boundary: still mid-tier
    assert resolve(vram=15.4).name == "ada-10-15gb"


def test_turing_low_vram_tier():
    profile = resolve(name="NVIDIA GeForce RTX 2070", cc=TURING, vram=8.0)
    assert profile.name == "turing-8-11gb"
    assert profile.eval_batch_cap == 4
    assert profile.checkpoint_modes == (True,)


def test_turing_12gb_underreporting_lands_in_mid_tier():
    profile = resolve(name="RTX 2080 Ti", cc=TURING, vram=11.99)
    assert profile.name == "turing-12-15gb"


def test_non_rtx_falls_to_compatibility():
    profile = resolve(name="NVIDIA A100-SXM4-40GB", cc=(8, 0), vram=40.0)
    assert profile.name == "compatibility"
    assert profile.is_compatibility_only is True


def test_laptop_falls_to_compatibility():
    profile = resolve(name="NVIDIA GeForce RTX 4080 Laptop GPU", vram=12.0)
    assert profile.name == "compatibility"


def test_below_vram_floor_falls_to_compatibility():
    profile = resolve(name="RTX 3080", cc=(8, 6), vram=8.0)  # ampere floor is 10GB
    assert profile.name == "compatibility"


# --- _get_gpu_peak_flops ---------------------------------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("NVIDIA GeForce RTX 4090", 330.3e12),
        ("NVIDIA GeForce RTX 4090 D", 280.0e12),
        ("NVIDIA GeForce RTX 4080", 242.5e12),
        ("NVIDIA GeForce RTX 4080 SUPER", 260.0e12),
        ("NVIDIA GeForce RTX 5090", 360.0e12),
        ("NVIDIA GeForce RTX 3060", 51.0e12),
        ("Totally Unknown GPU", None),
    ],
)
def test_get_gpu_peak_flops(name, expected):
    assert train._get_gpu_peak_flops(name) == expected


# --- autotune cache round-trip --------------------------------------------

def test_autotune_cache_round_trip(tmp_path):
    path = tmp_path / "cache" / "gpu-profile-v2.json"
    entries = {
        "RTX 4080|8.9|17171480576|2.13|Windows|2048": {
            "train_batch_size": 32,
            "use_activation_checkpointing": False,
            "tok_per_sec": 123456.789,
            "peak_memory_bytes": 14000000000,
            "updated_unix": 1752537600,
        }
    }
    train._save_autotune_entries(path, entries)
    assert path.exists()
    assert train._load_autotune_entries(path) == entries


def test_autotune_cache_missing_file_returns_empty(tmp_path):
    assert train._load_autotune_entries(tmp_path / "nope.json") == {}


def test_autotune_cache_corrupt_file_returns_empty(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{not json!!")
    assert train._load_autotune_entries(path) == {}


def test_autotune_cache_non_dict_payload_returns_empty(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]")
    assert train._load_autotune_entries(path) == {}
    path.write_text('{"entries": [1, 2]}')
    assert train._load_autotune_entries(path) == {}


# --- _filter_train_batch_sizes divisibility --------------------------------

def test_filter_train_batch_sizes_keeps_divisible_candidates():
    # TOTAL_BATCH_SIZE = 2**19, MAX_SEQ_LEN = 2048 -> batch must divide 256
    assert train._filter_train_batch_sizes((64, 32, 16, 8, 4)) == [64, 32, 16, 8, 4]


def test_filter_train_batch_sizes_drops_non_divisible_and_invalid():
    assert train._filter_train_batch_sizes((3, 5, 0, -2, 8)) == [8]


def test_filter_train_batch_sizes_dedups_preserving_order():
    assert train._filter_train_batch_sizes((8, 8, 4, 8)) == [8, 4]


def test_filter_train_batch_sizes_raises_when_nothing_valid():
    with pytest.raises(RuntimeError):
        train._filter_train_batch_sizes((3, 5, 0))


# --- evaluate_bpb math on a constant-loss stub model -----------------------

class _ConstantLossModel:
    """Stub model returning a constant per-token loss of ln(2) nats."""

    def __init__(self, loss_nats):
        self.loss_nats = loss_nats

    def __call__(self, x, y, reduction="none"):
        assert reduction == "none"
        return torch.full((y.numel(),), self.loss_nats, dtype=torch.float32)


def test_evaluate_bpb_constant_loss(monkeypatch):
    batch_size = 2
    seq_len = prepare.MAX_SEQ_LEN

    def fake_loader(tokenizer, bs, sl, split, device=None, dataset=None):
        assert split == "val"
        while True:
            x = torch.zeros((bs, sl), dtype=torch.long)
            y = torch.ones((bs, sl), dtype=torch.long)
            yield x, y, 0

    # every token id maps to 2 bytes
    token_bytes = torch.full((prepare.VOCAB_SIZE,), 2, dtype=torch.long)
    monkeypatch.setattr(prepare, "make_dataloader", fake_loader)
    monkeypatch.setattr(prepare, "get_token_bytes", lambda device=None, dataset=None: token_bytes)

    model = _ConstantLossModel(loss_nats=math.log(2))
    tokenizer = type("Tok", (), {"dataset": "tinystories"})()
    bpb = prepare.evaluate_bpb(
        model,
        tokenizer,
        batch_size,
        device="cpu",
        dataset="tinystories",
        eval_tokens=batch_size * seq_len,  # exactly one step
    )
    # loss ln(2) nats/token over 2 bytes/token -> 0.5 bits per byte
    assert bpb == pytest.approx(0.5, rel=1e-6)


def test_evaluate_bpb_excludes_zero_byte_tokens(monkeypatch):
    batch_size = 1
    seq_len = prepare.MAX_SEQ_LEN

    def fake_loader(tokenizer, bs, sl, split, device=None, dataset=None):
        while True:
            x = torch.zeros((bs, sl), dtype=torch.long)
            # half the targets are token id 0 (zero bytes: special), half id 1 (2 bytes)
            y = torch.arange(sl, dtype=torch.long).remainder(2).unsqueeze(0)
            yield x, y, 0

    token_bytes = torch.full((prepare.VOCAB_SIZE,), 2, dtype=torch.long)
    token_bytes[0] = 0
    monkeypatch.setattr(prepare, "make_dataloader", fake_loader)
    monkeypatch.setattr(prepare, "get_token_bytes", lambda device=None, dataset=None: token_bytes)

    model = _ConstantLossModel(loss_nats=math.log(2))
    tokenizer = type("Tok", (), {"dataset": "tinystories"})()
    bpb = prepare.evaluate_bpb(
        model, tokenizer, batch_size, device="cpu", dataset="tinystories",
        eval_tokens=batch_size * seq_len,
    )
    # only the 1024 two-byte tokens count: (1024 * ln2) / (ln2 * 2048) = 0.5
    assert bpb == pytest.approx(0.5, rel=1e-6)


# --- run artifact directory ------------------------------------------------

class _StubModel:
    def state_dict(self):
        return {"w": torch.zeros(1)}


def test_resolve_out_dir_defaults_to_cwd(monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_OUT_DIR", raising=False)
    assert train._resolve_out_dir() == "."


def test_resolve_out_dir_ignores_empty_env(monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_OUT_DIR", "")
    assert train._resolve_out_dir() == "."


def test_checkpoint_lands_in_out_dir_and_not_cwd(tmp_path, monkeypatch):
    # A containerized run must not drop a checkpoint into the checked-out repo.
    cwd = tmp_path / "work"
    out_dir = tmp_path / "out" / "nested"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("AUTORESEARCH_OUT_DIR", str(out_dir))

    train._save_pre_eval_checkpoint(_StubModel())

    assert (out_dir / "checkpoint_pre_eval.pt").exists()  # created the dir too
    assert list(cwd.iterdir()) == []


def test_checkpoint_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_OUT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    train._save_pre_eval_checkpoint(_StubModel())

    assert (tmp_path / "checkpoint_pre_eval.pt").exists()


# --- input-bound diagnostic ------------------------------------------------

def classify(data=0.0, gpu_wait=0.0, wall=1.0, steps=2):
    return train._classify_step_bound(data, gpu_wait, wall, steps)


def test_starved_loop_reads_as_input_bound():
    # CPU sits in the dataloader; the GPU drained the queue and never made
    # the CPU wait.
    result = classify(data=0.8, gpu_wait=0.05)
    assert result["verdict"] == "input-bound"
    assert result["data_fraction"] == pytest.approx(0.8)
    assert result["gpu_wait_fraction"] == pytest.approx(0.05)
    assert result["sampled_steps"] == 2


def test_gpu_blocked_loop_reads_as_compute_bound():
    assert classify(data=0.1, gpu_wait=0.7)["verdict"] == "compute-bound"


@pytest.mark.parametrize(
    "data,gpu_wait",
    [
        (0.6, 0.3),   # lots of dataloader time, but the CPU still waits on the GPU
        (0.3, 0.3),   # neither side dominates
        (0.5, 0.2),   # exactly on both thresholds -> refuses to call it
    ],
)
def test_ambiguous_splits_read_as_mixed(data, gpu_wait):
    assert classify(data=data, gpu_wait=gpu_wait)["verdict"] == "mixed"


def test_no_sampled_steps_returns_none():
    # A 1-step run samples nothing, because step 0 is always excluded.
    assert classify(data=0.8, gpu_wait=0.05, steps=0) is None


def test_zero_wall_time_returns_none():
    assert classify(data=0.0, gpu_wait=0.0, wall=0.0) is None


def test_fractions_are_relative_to_sampled_wall_time():
    result = classify(data=1.0, gpu_wait=0.1, wall=4.0, steps=3)
    assert result["data_fraction"] == pytest.approx(0.25)
    assert result["gpu_wait_fraction"] == pytest.approx(0.025)
    assert result["sampled_steps"] == 3
    assert result["verdict"] == "mixed"


# --- the residual column ---------------------------------------------------

def test_three_columns_account_for_the_whole_step():
    result = classify(data=0.3, gpu_wait=0.2)
    assert result["other_fraction"] == pytest.approx(0.5)
    total = result["data_fraction"] + result["gpu_wait_fraction"] + result["other_fraction"]
    assert total == pytest.approx(1.0)


def test_residual_is_clamped_to_zero_by_float_noise():
    # Both columns are timed sub-intervals of the same step, so they can only
    # exceed its wall clock by noise -- which must not print as a negative share.
    assert classify(data=0.5, gpu_wait=0.5 + 1e-12)["other_fraction"] == 0.0


def test_residual_does_not_move_the_verdict():
    # Same two timed columns, different amounts of unaccounted step time: the
    # residual is reported, never folded into either side of the threshold.
    starved = classify(data=0.8, gpu_wait=0.05)
    assert starved["verdict"] == "input-bound"
    assert starved["other_fraction"] == pytest.approx(0.15)

    diluted = classify(data=0.8, gpu_wait=0.05, wall=2.0)
    assert diluted["verdict"] == "mixed"
    assert diluted["other_fraction"] == pytest.approx(0.575)


def test_large_residual_is_what_makes_mixed_readable():
    # The likely shape of a first smoke run: the dataloader is busy and the CPU
    # rarely blocks on the device, yet most of the step is in neither column.
    # That remainder is kernel-enqueue time and possibly launch-queue
    # backpressure -- hidden GPU wait -- so "mixed" must not be read as
    # input-bound headroom.
    result = classify(data=0.35, gpu_wait=0.10)
    assert result["verdict"] == "mixed"
    assert result["other_fraction"] > result["data_fraction"]


# --- the diagnostic survives a failure after training ----------------------

def diagnostic(**kwargs):
    return train._format_input_bound_lines(classify(**kwargs))


def keyed(lines):
    return {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in lines}


def test_lines_carry_every_column_and_the_verdict():
    values = keyed(diagnostic(data=0.8, gpu_wait=0.05))
    assert values["dataloader_percent"] == "80.0"
    assert values["gpu_wait_percent"] == "5.0"
    assert values["other_percent"] == "15.0"
    assert values["loop_bound"] == "input-bound (over 2 step(s), step 0 excluded)"


def test_too_few_steps_says_so_rather_than_printing_a_number():
    # A run that sampled nothing must not render as 0.0% dataloader time --
    # that reads as a measured "not input-bound".
    values = keyed(train._format_input_bound_lines(None))
    assert set(values) == {
        "dataloader_percent",
        "gpu_wait_percent",
        "other_percent",
        "loop_bound",
    }
    assert values["dataloader_percent"] == "n/a"
    assert values["loop_bound"] == "n/a (needs at least 2 steps)"


def test_both_emit_points_render_identically():
    # main() prints these lines twice: once the moment training returns, once in
    # the final summary. A passing run must repeat itself exactly, so a reader
    # scraping loop_bound cannot find two disagreeing answers in one log.
    result = classify(data=0.35, gpu_wait=0.10)
    assert train._format_input_bound_lines(result) == train._format_input_bound_lines(result)


def test_sampled_step_count_travels_with_the_verdict():
    # The early emit point is the only place a timed-out run's numbers appear,
    # so the line has to say how few steps backed them without the summary's
    # context.
    line = keyed(diagnostic(data=0.8, gpu_wait=0.05, wall=4.0, steps=7))["loop_bound"]
    assert "over 7 step(s)" in line
    assert "step 0 excluded" in line


# --- loader device-wait attribution ----------------------------------------

def test_loader_wait_moves_from_data_to_gpu_wait():
    data, gpu_wait = train._split_loader_step_time(1.0, 0.25)
    assert data == pytest.approx(0.75)
    assert gpu_wait == pytest.approx(0.25)


def test_no_loader_wait_leaves_data_untouched():
    assert train._split_loader_step_time(1.0, 0.0) == (1.0, 0.0)


def test_float_noise_cannot_make_either_column_negative():
    # the wait is measured inside the data region, so a tiny negative delta is
    # noise, not a signal
    data, gpu_wait = train._split_loader_step_time(1.0, -1e-12)
    assert data == pytest.approx(1.0)
    assert gpu_wait == 0.0


def test_wait_is_clamped_to_the_time_it_was_measured_inside():
    data, gpu_wait = train._split_loader_step_time(0.5, 0.9)
    assert (data, gpu_wait) == (0.0, 0.5)


def test_split_conserves_total_step_time():
    data, gpu_wait = train._split_loader_step_time(2.0, 0.75)
    assert data + gpu_wait == pytest.approx(2.0)


def test_unattributed_loader_wait_would_invert_the_verdict():
    # A compute-bound step: next() measured 0.7s wall, but 0.6s of that was the
    # loader blocked on its own host-to-device copy, and .item() only had 0.1s
    # of queue left to drain. Charging the copy wait to data time reads as
    # "input-bound" -- the exact misdiagnosis _split_loader_step_time prevents.
    raw_data, item_wait, wall = 0.7, 0.1, 1.0
    assert classify(data=raw_data, gpu_wait=item_wait, wall=wall)["verdict"] == "input-bound"

    data, loader_wait = train._split_loader_step_time(raw_data, 0.6)
    result = classify(data=data, gpu_wait=loader_wait + item_wait, wall=wall)
    assert result["verdict"] == "compute-bound"


# --- dataloader CPU path ---------------------------------------------------

class _FixedLengthTokenizer:
    """Emits documents of exactly row_capacity tokens, each id used once.

    Unique ascending ids make batch-content bugs visible: a duplicated or
    half-overwritten batch shows up as a repeated or non-contiguous id run.
    """

    dataset = "tinystories"

    def __init__(self, doc_len):
        self.doc_len = doc_len
        self.next_id = 100

    def get_bos_token_id(self):
        return 0

    def encode(self, texts, prepend=None):
        encoded = []
        for _ in texts:
            encoded.append([prepend] + [self.next_id + i for i in range(self.doc_len - 1)])
            self.next_id += self.doc_len
        return encoded


@pytest.fixture
def cpu_loader(monkeypatch):
    def fake_document_batches(split, dataset=None, tokenizer_batch_size=128):
        while True:
            yield ["doc"] * 4, 1

    monkeypatch.setattr(prepare, "_document_batches", fake_document_batches)

    def build(B, T, stats=None):
        return prepare.make_dataloader(
            _FixedLengthTokenizer(doc_len=T + 1),
            B,
            T,
            "train",
            device="cpu",
            dataset="tinystories",
            buffer_size=8,
            stats=stats,
        )

    return build


def test_cpu_loader_packs_full_rows_and_shifts_targets(cpu_loader):
    loader = cpu_loader(B=2, T=4)
    inputs, targets, epoch = next(loader)
    # inputs/targets alias one reusable buffer, so a caller comparing batches
    # must copy -- true before this change and unchanged by it
    inputs, targets = inputs.clone(), targets.clone()

    assert inputs.shape == (2, 4)
    assert targets.shape == (2, 4)
    assert inputs.dtype == torch.long
    assert epoch == 1
    # every row starts with BOS and is one document, so targets is inputs shifted
    assert inputs[0].tolist() == [0, 100, 101, 102]
    assert targets[0].tolist() == [100, 101, 102, 103]
    assert inputs[1].tolist() == [0, 105, 106, 107]


def test_cpu_loader_yields_distinct_successive_batches(cpu_loader):
    loader = cpu_loader(B=2, T=4)
    first = next(loader)[0].clone()
    second = next(loader)[0].clone()
    assert not torch.equal(first, second)
    # documents are consumed in order, so batch 2 continues where batch 1 stopped
    assert second[0].tolist() == [0, 110, 111, 112]


def test_cpu_loader_records_no_device_wait(cpu_loader):
    # there is no host-to-device copy to wait on off the GPU, so the stats dict
    # must stay untouched rather than accumulating a bogus zero-wait entry
    stats = {"gpu_wait_seconds": 0.0}
    loader = cpu_loader(B=2, T=4, stats=stats)
    for _ in range(3):
        next(loader)
    assert stats == {"gpu_wait_seconds": 0.0}


def test_loader_accepts_no_stats(cpu_loader):
    loader = cpu_loader(B=2, T=4, stats=None)
    assert next(loader)[0].shape == (2, 4)


# --- misc runtime plumbing -------------------------------------------------

def test_runtime_config_has_no_use_compile_field():
    # constant-False use_compile plumbing was removed (audit finding 14)
    assert "use_compile" not in train.RuntimeConfig.__dataclass_fields__
    assert train.USE_COMPILE is False
