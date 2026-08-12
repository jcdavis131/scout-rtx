"""Tests for prepare.safe_torch_load / get_token_bytes.

safe_torch_load (added by the torch.load hardening fix) prefers the
restricted `weights_only=True` unpickler and only falls back to
`weights_only=False` when the artifact carries non-tensor objects the
restricted loader rejects. These tests exercise both paths against real
torch.load (no mocking of torch itself) so a regression that silently drops
the weights_only=True attempt, or that stops falling back for legitimate
mixed-content checkpoints, would be caught.
"""

import torch

import prepare


class _NotATensor:
    """Plain object (module-level so torch.save can pickle it) used to force
    the restricted weights_only=True unpickler to reject the payload."""

    def __init__(self, value):
        self.value = value


def test_safe_torch_load_plain_tensor_uses_weights_only_true(tmp_path, monkeypatch):
    path = tmp_path / "tensor.pt"
    torch.save(torch.tensor([1, 2, 3], dtype=torch.int32), path)

    calls = []
    real_torch_load = torch.load

    def recording_load(*args, **kwargs):
        calls.append(dict(kwargs))
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr(prepare.torch, "load", recording_load)

    with open(path, "rb") as f:
        result = prepare.safe_torch_load(f)

    assert torch.equal(result, torch.tensor([1, 2, 3], dtype=torch.int32))
    # A plain tensor-only artifact must succeed on the first (safe) attempt —
    # no silent fallback to weights_only=False for the common case.
    assert len(calls) == 1
    assert calls[0]["weights_only"] is True


def test_safe_torch_load_falls_back_for_non_tensor_payload(tmp_path, monkeypatch):
    """Regression test: a failed weights_only=True attempt on a *file handle*
    (the real call pattern used by get_token_bytes) partially consumes the
    handle. The fallback must rewind before retrying with weights_only=False
    or it reads from the wrong offset and raises instead of loading."""
    path = tmp_path / "mixed.pt"
    torch.save({"tensor": torch.arange(4), "extra": _NotATensor(42)}, path)

    calls = []
    real_torch_load = torch.load

    def recording_load(*args, **kwargs):
        calls.append(dict(kwargs))
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr(prepare.torch, "load", recording_load)

    with open(path, "rb") as f:
        result = prepare.safe_torch_load(f)

    assert torch.equal(result["tensor"], torch.arange(4))
    assert result["extra"].value == 42
    # First attempt is the restricted (safe) unpickler and it must be rejected
    # by the non-tensor payload; second attempt is the explicit fallback.
    assert [c["weights_only"] for c in calls] == [True, False]


def test_safe_torch_load_rewinds_plain_file_object_before_fallback(tmp_path):
    """Same as above but without mocking torch.load at all, so it also
    catches a regression in the real torch.load error-handling path."""
    path = tmp_path / "mixed_real.pt"
    torch.save({"tensor": torch.arange(3), "extra": _NotATensor("hi")}, path)

    with open(path, "rb") as f:
        result = prepare.safe_torch_load(f)

    assert torch.equal(result["tensor"], torch.arange(3))
    assert result["extra"].value == "hi"


def test_safe_torch_load_ignores_caller_supplied_weights_only_kwarg(tmp_path):
    """A caller passing weights_only explicitly must not bypass the helper's
    own weights_only=True-first policy — the kwarg is always re-derived."""
    path = tmp_path / "tensor.pt"
    torch.save(torch.tensor([7]), path)

    with open(path, "rb") as f:
        result = prepare.safe_torch_load(f, weights_only=False)
    assert torch.equal(result, torch.tensor([7]))


def test_get_token_bytes_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "DATASETS_DIR", tmp_path)
    tokenizer_dir = tmp_path / "tinystories" / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    expected = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    torch.save(expected, tokenizer_dir / "token_bytes.pt")

    result = prepare.get_token_bytes(device="cpu", dataset="tinystories")
    assert torch.equal(result, expected)
