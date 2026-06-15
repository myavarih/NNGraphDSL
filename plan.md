# NNGraph DSL — Implementation Plan

Mirrors the style of the EVM ANTLR project in `PycharmProjects/Antlr`.

---

## Tech Stack

- Python 3.9+
- ANTLR4 4.13+ → lexer/parser from `NNGraph.g4`
- `antlr4-python3-runtime`
- PyTorch 2.x (output target only — not a compiler dependency)

---

## Directory Layout

```
NNGraphDSL/
├── NNGraph.g4                         # ANTLR4 grammar
├── gen/                               # ANTLR-generated files (gitignored)
│   ├── NNGraphLexer.py
│   ├── NNGraphParser.py
│   ├── NNGraphListener.py
│   └── NNGraphVisitor.py
├── required_code_collection/          # mirrors EVM project
│   ├── ast.py                         # AST + TreeNode + traverse_ast
│   └── make_ast_subtree.py            # attaches ctx children to AST
├── custom_listener.py                 # semantic analysis via Listener
├── code_generator.py                  # CodeGenerator class
├── main.py                            # CLI entry point
├── input/                             # .nng sample files
│   ├── mlp.nng
│   ├── resblock.nng
│   ├── inception.nng
│   └── transformer.nng
├── output/                            # generated .py files land here
└── requirements.txt
```

---

## Phase 1 — Bootstrap

**Goal:** repo skeleton + ANTLR pipeline running end-to-end on a stub grammar.

Steps:
1. Create directory layout above.
2. `requirements.txt`: `antlr4-python3-runtime==4.13.*`
3. Download `antlr-4.13.1-complete.jar` to repo root.
4. Write minimal stub `NNGraph.g4` (just `model` block) — enough to prove the pipeline.
5. Generate:
   ```bash
   java -jar antlr-4.13.1-complete.jar -Dlanguage=Python3 -listener -o gen/ NNGraph.g4
   ```
6. Copy `required_code_collection/ast.py` and `make_ast_subtree.py` from EVM project (identical contract — `AST`, `TreeNode`, `traverse_ast`, `make_ast_subtree`).
7. Write `main.py` skeleton (see Phase 5 for final form) — parse file, walk with empty listener, print parse tree.

Deliverable: `python main.py -i input/mlp.nng` prints a parse tree without crashing.

---

## Phase 2 — Grammar (`NNGraph.g4`)

**Goal:** complete grammar covering all constructs in the spec.

Grammar conventions (same as `evm.g4`):
- Parser rules: `snake_case`
- Lexer tokens: `ALL_CAPS`
- Whitespace + comments: `-> skip`
- Lexer rules at bottom of file

```antlr
grammar NNGraph;

// ── Parser Rules ──────────────────────────────────────────────

program: model_block graph_block config_block? EOF;

model_block: MODEL ID '{' input_decl output_decl '}';
input_decl:  INPUT ID ':' TENSOR '(' shape_expr ')';
output_decl: OUTPUT ID;

graph_block: GRAPH '{' (node_decl | edge_decl)* '}';

node_decl: NODE ID ':' layer_expr;
layer_expr: ID '(' param_list? ')';
param_list: param (',' param)*;
param:      ID '=' value;

edge_decl: EDGE ID ARROW ID ('[' LABEL '=' STRING ']')?;

config_block: CONFIG '{' config_entry* '}';
config_entry: ID '=' value;

value:      INT_LITERAL
          | FLOAT_LITERAL
          | BOOL_LITERAL
          | STRING
          | NONE
          | shape_expr;

shape_expr: '(' INT_LITERAL (',' INT_LITERAL)* ')';

// ── Lexer Tokens ──────────────────────────────────────────────

MODEL:  'model';
GRAPH:  'graph';
CONFIG: 'config';
NODE:   'node';
EDGE:   'edge';
INPUT:  'input';
OUTPUT: 'output';
TENSOR: 'tensor';
LABEL:  'label';
NONE:   'None';

BOOL_LITERAL: 'true' | 'false';
ARROW:        '->';

ID:           [a-zA-Z_][a-zA-Z0-9_]*;
INT_LITERAL:  [0-9]+;
FLOAT_LITERAL:[0-9]+ '.' [0-9]+;
STRING:       '"' (~["\r\n])* '"';

WS:           [ \t]+ -> skip;
NEWLINE:      [\r\n]+ -> skip;
LINE_COMMENT: '//' ~[\r\n]* -> skip;
BLOCK_COMMENT:'/*' .*? '*/' -> skip;
```

Deliverable: grammar parses all 4 spec examples without ANTLR errors.

---

## Phase 3 — Semantic Analysis (`custom_listener.py`)

**Goal:** `NNGraphCustomListener` subclasses `NNGraphListener`, uses `exit*` methods and the global `error()` pattern from EVM project. Builds an internal graph model and enforces all 8 spec rules.

### Error Helper (identical pattern to EVM)

```python
def error(msg, ctx):
    line = ctx.start.line
    col  = ctx.start.column
    print(f"[line {line}:{col}] SemanticError: {msg}")
    raise SystemExit(1)

def warn(msg, ctx):
    line = ctx.start.line
    print(f"[line {line}] Warning: {msg}")
```

### Listener Class

```python
class NNGraphCustomListener(NNGraphListener):
    def __init__(self):
        self.model_name   = None
        self.input_id     = None
        self.input_shape  = None
        self.output_id    = None
        self.nodes        = {}   # id → {"type": str, "params": dict, "line": int}
        self.edges        = []   # list of {"src", "dst", "label"}
        self.config       = {}

        self.ast          = AST()
        self.rule_names   = []
```

### `exit*` Methods and What They Do

| Method | Action |
|---|---|
| `exitInput_decl` | store `self.input_id`, `self.input_shape`; add input as pseudo-node |
| `exitOutput_decl` | store `self.output_id` |
| `exitNode_decl` | parse layer type + params; add to `self.nodes`; error on duplicate ID |
| `exitParam` | attach `.param_name` and `.param_value` to ctx; validate type via `LAYER_SCHEMA` |
| `exitEdge_decl` | add to `self.edges`; error if src/dst not in `self.nodes` |
| `exitConfig_entry` | store in `self.config` |
| `exitGraph_block` | run post-graph validation passes (see below) |
| `exitEveryRule` | call `make_ast_subtree(self.ast, ctx, rule_name)` (same as EVM) |

### Post-Graph Validation (in `exitGraph_block`)

Run in order:

1. **Undefined references** — both `src` and `dst` of every edge must exist in `self.nodes`.
2. **Unique node IDs** — already enforced in `exitNode_decl`.
3. **Orphan nodes** — nodes not in any edge → `warn()`.
4. **Residual arity** — count in-edges per node; `Residual` must have exactly 2 → `error()`.
5. **Cycle detection** — Kahn's topological sort; if queue empties with remaining nodes → `error()`.
6. **Input reachability** — BFS from `input_id`; unreachable nodes → `warn()`.
7. **Output reachability** — `output_id` must be reachable → `error()`.

### Layer Param Schema (for `exitParam` type checking)

```python
LAYER_SCHEMA = {
    "Linear":        {"in_features": int, "out_features": int, "bias": bool},
    "Conv2d":        {"in_ch": int, "out_ch": int, "kernel": int, "stride": int, "padding": int},
    "Conv1d":        {"in_ch": int, "out_ch": int, "kernel": int, "stride": int},
    "BatchNorm2d":   {"num_features": int},
    "LayerNorm":     {"normalized_shape": (int, tuple)},
    "MaxPool2d":     {"kernel": int, "stride": int},
    "AvgPool2d":     {"kernel": int, "stride": int},
    "Dropout":       {"p": float},
    "Flatten":       {"start_dim": int, "end_dim": int},
    "Embedding":     {"num_embeddings": int, "embedding_dim": int},
    "MultiHeadAttn": {"embed_dim": int, "num_heads": int},
    "LSTM":          {"input_size": int, "hidden_size": int, "num_layers": int},
    "GRU":           {"input_size": int, "hidden_size": int},
    "ReLU": {}, "Sigmoid": {}, "Tanh": {}, "GELU": {},
    "Softmax":    {"dim": int},
    "LeakyReLU":  {"negative_slope": float},
    "ELU":        {"alpha": float},
    "Add": {}, "Concat": {"dim": int}, "Residual": {},
    "Split":      {"chunks": int, "dim": int},
}
```

Deliverable: `walker.walk(listener, parse_tree)` raises SystemExit with correct line on each of the 8 error conditions; passes cleanly on valid inputs.

---

## Phase 4 — Code Generator (`code_generator.py`)

**Goal:** `CodeGenerator.generate(traversal)` receives the post-order AST traversal (same contract as EVM's `ast.traverse_ast`) and returns a Python source string.

Unlike EVM (which emits bytecode via operand/code stacks), NNGraph emits structured Python — so the generator reads the traversal to reconstruct the graph model, then does a second topological pass to emit code.

### Class Structure

```python
class CodeGenerator:
    def __init__(self):
        self.model_name  = None
        self.input_id    = None
        self.input_shape = None
        self.output_id   = None
        self.nodes       = {}   # id → {"type": str, "params": dict}
        self.edges       = []
        self.config      = {}

    def generate(self, traversal):
        self._reconstruct(traversal)
        topo_order = self._topo_sort()
        in_degree, out_degree = self._compute_degrees()
        init_lines    = self._emit_init(topo_order)
        forward_lines = self._emit_forward(topo_order, in_degree, out_degree)
        main_block    = self._emit_main()
        return self._assemble(init_lines, forward_lines, main_block)
```

### `_emit_init` — `__init__` method

One line per node that maps to an `nn.Module`. Nodes without a module (`Add`, `Concat`, `Residual`, `Split`) are skipped.

PyTorch param name mapping:

| DSL param | PyTorch param |
|---|---|
| `in_ch` | `in_channels` |
| `out_ch` | `out_channels` |
| `kernel` | `kernel_size` |
| `MultiHeadAttn` | add `batch_first=True` |

### `_emit_forward` — `forward()` method

Walk `topo_order`. For each node, determine its variable name(s) and emit the appropriate pattern:

| Node type | Emitted pattern |
|---|---|
| regular layer (1→1) | `x = self.<id>(x)` |
| fan-out node | assign named branch vars: `<dst_id> = self.<id>(x)` per outgoing edge |
| `Add` | `x = <a> + <b>` (or `sum([...])` for >2) |
| `Concat(dim=d)` | `x = torch.cat([<a>, <b>], dim=<d>)` |
| `Residual` | detect shortcut edge by label; emit `identity = <shortcut>` before main path starts, then `x = x + identity` at the Residual node |
| `Split(chunks=n, dim=d)` | `parts = torch.chunk(x, n, dim=d)` |
| `MultiHeadAttn` | `<id>_out, _ = self.<id>(x, x, x)` |
| `LSTM` / `GRU` | `x, _ = self.<id>(x)` |
| LayerNorm with 2 in-edges | `x = self.<id>(<branch> + <skip>)` — transformer residual pattern |

### `_emit_main` — `if __name__ == '__main__':` block

Uses `self.config` for `device` and `batch_size`; uses `self.input_shape` for `torch.randn`.

Deliverable: `code_generator.generate(traversal)` output matches spec examples (whitespace-normalized).

---

## Phase 5 — Entry Point (`main.py`)

Same structure as EVM `main.py`:

```python
from antlr4 import *
import argparse
from gen.NNGraphLexer import NNGraphLexer
from gen.NNGraphParser import NNGraphParser
from custom_listener import NNGraphCustomListener
from code_generator import CodeGenerator
from required_code_collection.ast_to_networkx_graph import show_ast


def main(arguments):
    stream       = FileStream(arguments.input, encoding='utf8')
    lexer        = NNGraphLexer(stream)
    token_stream = CommonTokenStream(lexer)
    parser       = NNGraphParser(token_stream)
    parse_tree   = parser.program()

    listener = NNGraphCustomListener()
    listener.rule_names = parser.ruleNames
    walker = ParseTreeWalker()
    walker.walk(listener, parse_tree)

    ast       = listener.ast
    traversal = ast.traverse_ast(ast.root)

    code_gen   = CodeGenerator()
    final_code = code_gen.generate(traversal)
    print(final_code)

    out_path = arguments.output
    with open(out_path, 'w') as f:
        f.write(final_code)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input',  default='input/mlp.nng')
    ap.add_argument('-o', '--output', default='output/model.py')
    main(ap.parse_args())
```

Usage:
```bash
python main.py -i input/mlp.nng -o output/mlp.py
python -c "import sys; sys.path.insert(0,'output'); import mlp; print('OK')"
```

---

## Phase 6 — Tests

| File | Tests |
|---|---|
| `tests/test_parser.py` | Parse all 4 `.nng` examples; assert `parser.getNumberOfSyntaxErrors() == 0` |
| `tests/test_listener.py` | Valid graphs walk cleanly; each of 8 error conditions triggers `SystemExit` with correct line in stderr |
| `tests/test_codegen.py` | `generate(traversal)` for each example matches expected output (whitespace-normalized) |
| `tests/test_e2e.py` | Full `main.py` pipeline: `.nng` → `.py` → `python -c "import <module>"` exits 0 |

Run: `python -m pytest tests/ -v`

---

## Implementation Order

```
Phase 1  bootstrap + AST copy          1 session
Phase 2  full grammar                  1 session
Phase 3  custom_listener.py            1 session
Phase 4  code_generator.py             1–2 sessions
Phase 5  main.py (wire up)             same session as Phase 4
Phase 6  tests                         1 session
```

---

## Known Hard Problems

1. **Residual shortcut detection** — identify which in-edge has `[label="shortcut"]`; save `identity` before the main conv path begins. Wrong order → wrong forward.
2. **Branch variable tracking** — fan-out nodes need per-downstream named vars. Must track which variable name is "live" at each consumer.
3. **MultiHeadAttn triple-pass** — always emits `self.attn(x, x, x)` for self-attention; no way to express cross-attention in the DSL yet.
4. **Transformer implicit residual** — `LayerNorm` with 2 in-edges (one labeled `residual_*`) must generate `x = self.norm(x + skip)` without a `Residual()` node. Requires label inspection in codegen, not just type checking.
5. **AST traversal contract** — `make_ast_subtree` and `traverse_ast` produce a flat post-order list; the codegen `_reconstruct` pass must correctly rebuild the graph from that sequence. May need sentinel tokens (e.g. `node_decl`, `edge_decl`) in the traversal similar to EVM's operator list.
