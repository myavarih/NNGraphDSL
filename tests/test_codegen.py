"""Code-generator tests — verify structure and key patterns of emitted PyTorch code."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import compile_nng

INPUTS = os.path.join(os.path.dirname(__file__), "..", "input")


def _code(name):
    return compile_nng(os.path.join(INPUTS, f"{name}.nng"))


# ── MLP ───────────────────────────────────────────────────────────────────────

def test_mlp_class_declaration():
    code = _code("mlp")
    assert "class MLP(nn.Module):" in code


def test_mlp_linear_layers():
    code = _code("mlp")
    assert "nn.Linear(in_features=784, out_features=256)" in code
    assert "nn.Linear(in_features=256, out_features=128)" in code
    assert "nn.Linear(in_features=128, out_features=10)" in code


def test_mlp_dropout():
    code = _code("mlp")
    assert "nn.Dropout(p=0.3)" in code


def test_mlp_softmax():
    code = _code("mlp")
    assert "nn.Softmax(dim=1)" in code


def test_mlp_forward_linear_chain():
    code = _code("mlp")
    forward = code[code.index("def forward"):]
    # pure linear: all assignments in forward should reuse 'x'
    forward_lines = [l.strip() for l in forward.splitlines() if "self." in l and "def " not in l]
    for line in forward_lines:
        if "return" in line:
            continue
        assert line.startswith("x ="), f"Non-linear assignment in MLP forward: {line}"


def test_mlp_main_block():
    code = _code("mlp")
    assert "torch.device('cuda')" in code
    assert "torch.randn(64, 784)" in code


# ── ResBlock ─────────────────────────────────────────────────────────────────

def test_resblock_conv2d_params():
    code = _code("resblock")
    assert "nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)" in code


def test_resblock_no_residual_module():
    code = _code("resblock")
    # Residual() is a no-module node; must NOT appear as self.skip = ...
    assert "self.skip" not in code


def test_resblock_residual_addition():
    code = _code("resblock")
    forward = code[code.index("def forward"):]
    # some form of element-wise addition must appear
    assert "+" in forward


def test_resblock_flatten():
    code = _code("resblock")
    assert "nn.Flatten()" in code


# ── InceptionBlock ────────────────────────────────────────────────────────────

def test_inception_two_conv_branches():
    code = _code("inception")
    assert "self.convA1" in code
    assert "self.convB1" in code
    assert "self.convB2" in code


def test_inception_concat():
    code = _code("inception")
    forward = code[code.index("def forward"):]
    assert "torch.cat([" in forward
    assert "dim=1" in forward


def test_inception_no_concat_module():
    code = _code("inception")
    assert "self.cat1" not in code


def test_inception_batchnorm():
    code = _code("inception")
    assert "nn.BatchNorm2d(num_features=48)" in code


def test_inception_branch_vars_distinct():
    code = _code("inception")
    forward = code[code.index("def forward"):]
    # the two branch vars must be different — convA1 and convB1 lines both present
    assert "convA1" in forward
    assert "convB1" in forward


# ── TransformerEncoder ────────────────────────────────────────────────────────

def test_transformer_multihead_attention():
    code = _code("transformer")
    assert "nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)" in code


def test_transformer_attn_triple_pass():
    code = _code("transformer")
    forward = code[code.index("def forward"):]
    # self-attention: query == key == value
    lines = [l for l in forward.splitlines() if "self.attn(" in l]
    assert len(lines) == 1
    call = lines[0]
    # extract the three args inside self.attn(...)
    args_str = call[call.index("self.attn(") + len("self.attn("):call.rindex(")")]
    args = [a.strip() for a in args_str.split(",")]
    assert len(args) == 3
    assert args[0] == args[1] == args[2], f"Attention args not equal: {args}"


def test_transformer_layernorm():
    code = _code("transformer")
    assert "nn.LayerNorm(normalized_shape=128)" in code


def test_transformer_implicit_residual_norm1():
    code = _code("transformer")
    forward = code[code.index("def forward"):]
    # norm1 must receive a sum of two tensors inline
    norm1_lines = [l.strip() for l in forward.splitlines() if "self.norm1(" in l]
    assert len(norm1_lines) == 1
    assert "+" in norm1_lines[0]


def test_transformer_implicit_residual_norm2():
    code = _code("transformer")
    forward = code[code.index("def forward"):]
    norm2_lines = [l.strip() for l in forward.splitlines() if "self.norm2(" in l]
    assert len(norm2_lines) == 1
    assert "+" in norm2_lines[0]


def test_transformer_attn_tuple_unpack():
    code = _code("transformer")
    forward = code[code.index("def forward"):]
    attn_lines = [l for l in forward.splitlines() if "self.attn(" in l]
    assert ", _" in attn_lines[0]


def test_transformer_flatten():
    code = _code("transformer")
    assert "nn.Flatten(start_dim=0, end_dim=1)" in code
