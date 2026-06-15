"""End-to-end tests — .nng → .py → instantiate model → forward pass with torch."""
import os
import sys
import importlib.util
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import compile_nng

INPUTS  = os.path.join(os.path.dirname(__file__), "..", "input")
OUTPUTS = os.path.join(os.path.dirname(__file__), "..", "output")


def _load_model(nng_name, cls_name):
    src_path = os.path.join(INPUTS,  f"{nng_name}.nng")
    out_path = os.path.join(OUTPUTS, f"{nng_name}_test.py")
    compile_nng(src_path, output_path=out_path)
    spec = importlib.util.spec_from_file_location(nng_name, out_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)()


# ── MLP ───────────────────────────────────────────────────────────────────────

def test_mlp_forward_shape():
    model = _load_model("mlp", "MLP")
    x     = torch.randn(4, 784)
    out   = model(x)
    assert out.shape == (4, 10)


def test_mlp_output_sums_to_one():
    model = _load_model("mlp", "MLP")
    x     = torch.randn(4, 784)
    out   = model(x)
    # Softmax rows sum to 1
    sums  = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)


# ── ResBlock ─────────────────────────────────────────────────────────────────

def test_resblock_forward_shape():
    model = _load_model("resblock", "ResBlock")
    x     = torch.randn(2, 64, 32, 32)
    out   = model(x)
    # Flatten of (2, 64, 32, 32) → (2, 64*32*32)
    assert out.shape == (2, 64 * 32 * 32)


def test_resblock_deterministic_eval():
    model = _load_model("resblock", "ResBlock")
    model.eval()
    x    = torch.randn(1, 64, 32, 32)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)


# ── InceptionBlock ────────────────────────────────────────────────────────────

def test_inception_forward_shape():
    model = _load_model("inception", "InceptionBlock")
    x     = torch.randn(2, 32, 56, 56)
    out   = model(x)
    # Branch A: 16ch, Branch B: 32ch, concat → 48ch; H,W unchanged
    assert out.shape == (2, 48, 56, 56)


def test_inception_two_inputs_same_output_shape():
    model = _load_model("inception", "InceptionBlock")
    x1 = torch.randn(1, 32, 56, 56)
    x2 = torch.randn(3, 32, 56, 56)
    assert model(x1).shape[1:] == model(x2).shape[1:]


# ── TransformerEncoder ────────────────────────────────────────────────────────

def test_transformer_forward_shape():
    model = _load_model("transformer", "TransformerEncoder")
    x     = torch.randn(2, 512, 128)
    out   = model(x)
    # Flatten(start_dim=0, end_dim=1): (2,512,128) → (1024,128)
    assert out.shape == (1024, 128)


def test_transformer_eval_deterministic():
    model = _load_model("transformer", "TransformerEncoder")
    model.eval()
    x    = torch.randn(1, 512, 128)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)


# ── CLI: check-only flag ──────────────────────────────────────────────────────

def test_check_only_returns_none():
    path   = os.path.join(INPUTS, "mlp.nng")
    result = compile_nng(path, check_only=True)
    assert result is None


def test_check_only_no_file_written(tmp_path):
    out = str(tmp_path / "should_not_exist.py")
    compile_nng(os.path.join(INPUTS, "mlp.nng"), output_path=out, check_only=True)
    assert not os.path.exists(out)


# ── CLI: bad input raises SystemExit ─────────────────────────────────────────

def test_bad_input_exits():
    with pytest.raises((SystemExit, Exception)):
        compile_nng("/nonexistent/path.nng")
