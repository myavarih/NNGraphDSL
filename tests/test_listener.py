"""Semantic analyser tests — valid graphs pass; each error condition raises SystemExit."""
import os
import sys
import textwrap
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from antlr4 import FileStream, CommonTokenStream, ParseTreeWalker
from gen.NNGraphLexer import NNGraphLexer
from gen.NNGraphParser import NNGraphParser
from custom_listener import NNGraphCustomListener


def _walk(source: str):
    """Parse and walk source string; return listener. Raises SystemExit on semantic error."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".nng", delete=False) as f:
        f.write(textwrap.dedent(source))
        path = f.name
    try:
        stream   = FileStream(path, encoding="utf8")
        lexer    = NNGraphLexer(stream)
        parser   = NNGraphParser(CommonTokenStream(lexer))
        tree     = parser.program()
        listener = NNGraphCustomListener()
        listener.rule_names = parser.ruleNames
        ParseTreeWalker().walk(listener, tree)
        return listener
    finally:
        os.unlink(path)


VALID_LINEAR = """
    model M { input x : tensor(1) output out }
    graph {
        node fc : Linear(in_features=1, out_features=1)
        node out : ReLU()
        edge x -> fc
        edge fc -> out
    }
"""

# ── valid graphs pass ──────────────────────────────────────────────────────────

def test_valid_linear_passes():
    l = _walk(VALID_LINEAR)
    assert l.model_name == "M"
    assert l.output_id == "out"


def test_valid_resblock_passes():
    from main import compile_nng
    path = os.path.join(os.path.dirname(__file__), "..", "input", "resblock.nng")
    assert compile_nng(path, check_only=True) is None


def test_valid_inception_passes():
    from main import compile_nng
    path = os.path.join(os.path.dirname(__file__), "..", "input", "inception.nng")
    assert compile_nng(path, check_only=True) is None


def test_valid_transformer_passes():
    from main import compile_nng
    path = os.path.join(os.path.dirname(__file__), "..", "input", "transformer.nng")
    assert compile_nng(path, check_only=True) is None


# ── error: undefined edge reference ──────────────────────────────────────────

def test_undefined_node_in_edge():
    with pytest.raises(SystemExit):
        _walk("""
            model M { input x : tensor(1) output out }
            graph {
                node out : ReLU()
                edge x -> ghost
                edge ghost -> out
            }
        """)


# ── error: duplicate node ID ──────────────────────────────────────────────────

def test_duplicate_node_id():
    with pytest.raises(SystemExit):
        _walk("""
            model M { input x : tensor(1) output out }
            graph {
                node fc : Linear(in_features=1, out_features=1)
                node fc : Linear(in_features=1, out_features=2)
                node out : ReLU()
                edge x -> fc
                edge fc -> out
            }
        """)


# ── error: residual arity != 2 ────────────────────────────────────────────────

def test_residual_three_inputs():
    with pytest.raises(SystemExit):
        _walk("""
            model M { input x : tensor(1) output out }
            graph {
                node a : Linear(in_features=1, out_features=1)
                node b : Linear(in_features=1, out_features=1)
                node c : Linear(in_features=1, out_features=1)
                node skip : Residual()
                node out : ReLU()
                edge x -> a
                edge x -> b
                edge x -> c
                edge a -> skip
                edge b -> skip
                edge c -> skip
                edge skip -> out
            }
        """)


def test_residual_one_input():
    with pytest.raises(SystemExit):
        _walk("""
            model M { input x : tensor(1) output out }
            graph {
                node skip : Residual()
                node out : ReLU()
                edge x -> skip
                edge skip -> out
            }
        """)


# ── error: cycle ──────────────────────────────────────────────────────────────

def test_cycle_detected():
    with pytest.raises(SystemExit):
        _walk("""
            model M { input x : tensor(1) output out }
            graph {
                node a : Linear(in_features=1, out_features=1)
                node b : Linear(in_features=1, out_features=1)
                node out : ReLU()
                edge x -> a
                edge a -> b
                edge b -> a
                edge b -> out
            }
        """)


# ── error: param type mismatch ────────────────────────────────────────────────

def test_param_type_int_where_float():
    with pytest.raises(SystemExit):
        _walk("""
            model M { input x : tensor(1) output out }
            graph {
                node d : Dropout(p=1)
                node out : ReLU()
                edge x -> d
                edge d -> out
            }
        """)


def test_param_type_float_where_int():
    with pytest.raises(SystemExit):
        _walk("""
            model M { input x : tensor(1) output out }
            graph {
                node fc : Linear(in_features=1.0, out_features=1)
                node out : ReLU()
                edge x -> fc
                edge fc -> out
            }
        """)


# ── error: unknown layer type ────────────────────────────────────────────────

def test_unknown_layer():
    with pytest.raises(SystemExit):
        _walk("""
            model M { input x : tensor(1) output out }
            graph {
                node foo : BigBertLayer()
                node out : ReLU()
                edge x -> foo
                edge foo -> out
            }
        """)


# ── error: output node not declared in graph ─────────────────────────────────

def test_output_not_declared():
    with pytest.raises(SystemExit):
        _walk("""
            model M { input x : tensor(1) output missing }
            graph {
                node fc : Linear(in_features=1, out_features=1)
                edge x -> fc
            }
        """)


# ── error: orphan node (declared but never referenced in any edge) ────────────

def test_orphan_node_raises():
    with pytest.raises(SystemExit):
        _walk("""
            model M { input x : tensor(1) output out }
            graph {
                node orphan : ReLU()
                node out : ReLU()
                edge x -> out
            }
        """)
