"""Parser tests — verify all spec examples parse with zero syntax errors."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from antlr4 import FileStream, CommonTokenStream
from gen.NNGraphLexer import NNGraphLexer
from gen.NNGraphParser import NNGraphParser

INPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "input")


def _parse(path):
    stream = FileStream(path, encoding="utf8")
    lexer  = NNGraphLexer(stream)
    parser = NNGraphParser(CommonTokenStream(lexer))
    tree   = parser.start()
    return parser, tree


@pytest.mark.parametrize("filename", ["mlp.nng", "resblock.nng", "inception.nng", "transformer.nng"])
def test_no_syntax_errors(filename):
    parser, _ = _parse(os.path.join(INPUTS_DIR, filename))
    assert parser.getNumberOfSyntaxErrors() == 0


def test_model_name_mlp():
    from antlr4 import ParseTreeWalker
    from custom_listener import NNGraphCustomListener
    parser, tree = _parse(os.path.join(INPUTS_DIR, "mlp.nng"))
    listener = NNGraphCustomListener()
    listener.rule_names = parser.ruleNames
    ParseTreeWalker().walk(listener, tree)
    assert listener.model_name == "MLP"


def test_input_shape_resblock():
    from antlr4 import ParseTreeWalker
    from custom_listener import NNGraphCustomListener
    parser, tree = _parse(os.path.join(INPUTS_DIR, "resblock.nng"))
    listener = NNGraphCustomListener()
    listener.rule_names = parser.ruleNames
    ParseTreeWalker().walk(listener, tree)
    assert listener.input_shape == (64, 32, 32)


def test_config_parsed():
    from antlr4 import ParseTreeWalker
    from custom_listener import NNGraphCustomListener
    parser, tree = _parse(os.path.join(INPUTS_DIR, "mlp.nng"))
    listener = NNGraphCustomListener()
    listener.rule_names = parser.ruleNames
    ParseTreeWalker().walk(listener, tree)
    assert listener.config["batch_size"] == 64
    assert listener.config["device"] == "cuda"


def test_labeled_edge_parsed():
    from antlr4 import ParseTreeWalker
    from custom_listener import NNGraphCustomListener
    parser, tree = _parse(os.path.join(INPUTS_DIR, "resblock.nng"))
    listener = NNGraphCustomListener()
    listener.rule_names = parser.ruleNames
    ParseTreeWalker().walk(listener, tree)
    labels = [e["label"] for e in listener.edges if e["label"]]
    assert "shortcut" in labels
