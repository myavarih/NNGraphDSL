# NNGraph DSL — Report Notes

All technical facts needed for a LaTeX report. Organized by typical report section.
Pull numbers and code fragments directly from here; everything is verified against
the actual source files.

---

## 1. Introduction / Motivation

**Problem statement:**
Writing PyTorch `nn.Module` subclasses by hand is repetitive. A 10-layer network
requires a matching `__init__` declaration for every layer and a `forward()` that
threads tensors through them in the right order. For branching topologies (residual
blocks, inception-style parallel branches, transformer sub-layers), the programmer
must also track intermediate variable names manually and ensure no tensor is
overwritten while another branch still holds a reference to it.

**Goal of NNGraph DSL:**
Express a neural architecture as a directed graph; compile it to executable PyTorch.
The source file is the single source of truth for the architecture. The compiler
generates all boilerplate.

**Inspiration:**
NADER framework (Neural Architecture Design via Representation), which models neural
network architectures as directed computational graphs.

**Compiler target:**
PyTorch 2.10 (`nn.Module` subclass with `__init__` and `forward`).

---

## 2. Related Work / Background

**ANTLR4 (ANother Tool for Language Recognition):**
- Generates lexer and parser from a grammar file (`.g4`).
- Used here to produce Python 3 code: `NNGraphLexer.py`, `NNGraphParser.py`,
  `NNGraphListener.py`.
- Version: ANTLR 4.13.2 (`/usr/share/java/antlr-4.13.2-complete.jar`).
- Runtime: `antlr4-python3-runtime==4.13.*`.

**Listener pattern vs visitor pattern:**
ANTLR4 supports two tree-walking patterns. This compiler uses the Listener pattern:
`ParseTreeWalker` calls `enter*`/`exit*` methods automatically as it traverses the
tree. The EVM ANTLR project in `PycharmProjects/Antlr` uses the same pattern, and
this project follows the same code structure (`custom_listener.py` mirrors
`evm/custom_listener.py`).

**PyTorch `nn.Module`:**
The generated code subclasses `nn.Module`. Every parameterized layer becomes an
attribute of `self` in `__init__`. The `forward()` method calls them in topological
order.

---

## 3. Language Design

### 3.1 Program structure

Three top-level blocks:

```
model <Name> { input <id> : tensor(<shape>)  output <id> }
graph        { node declarations; edge declarations }
config       { batch_size = <int>; device = "cpu"|"cuda" }   (optional)
```

### 3.2 Grammar statistics

- Grammar file: `NNGraph.g4` (56 lines)
- Parser rules (14): `program`, `model_block`, `input_decl`, `output_decl`,
  `graph_block`, `node_decl`, `layer_expr`, `param_list`, `param`, `edge_decl`,
  `config_block`, `config_entry`, `value`, `shape_expr`
- Lexer tokens (20): keywords (`MODEL`, `GRAPH`, `CONFIG`, `NODE`, `EDGE`, `INPUT`,
  `OUTPUT`, `TENSOR`, `LABEL`, `NONE`), literals (`INT_LITERAL`, `FLOAT_LITERAL`,
  `STRING`, `BOOL_LITERAL`), operators (`ARROW`, `ID`), skipped tokens
  (`WS`, `NEWLINE`, `LINE_COMMENT`, `BLOCK_COMMENT`)

### 3.3 Types

| DSL type | Python type | Example |
|---|---|---|
| `int` | `int` | `128` |
| `float` | `float` | `0.1` |
| `bool` | `bool` | `true` / `false` |
| `string` | `str` | `"relu"` |
| `shape` | `tuple[int, ...]` | `(64, 32, 32)` |
| `None` | `NoneType` | `None` |

### 3.4 Layer types

24 layer types total: 13 parameterized layers + 7 activations + 4 graph operations.

Parameterized (have an `nn.Module` in `__init__`):
`Linear`, `Conv2d`, `Conv1d`, `BatchNorm2d`, `LayerNorm`, `MaxPool2d`, `AvgPool2d`,
`Dropout`, `Flatten`, `Embedding`, `MultiHeadAttn`, `LSTM`, `GRU`

Activations (have an `nn.Module` in `__init__`, no required parameters):
`ReLU`, `Sigmoid`, `Tanh`, `GELU`, `Softmax`, `LeakyReLU`, `ELU`

Graph operations (no `nn.Module` in `__init__`):
`Add`, `Concat`, `Residual`, `Split`

### 3.5 Edge syntax

Simple: `edge src -> dst`
Labeled: `edge src -> dst [label="name"]`

Labels serve two roles:
- `"shortcut"` marks the skip connection into a `Residual()` node.
- `"residual_*"` (any label starting with `residual`) marks a skip connection
  into a regular node with two in-edges (transformer pattern).

---

## 4. Implementation

### 4.1 Pipeline

```
.nng source
    │
    ▼  FileStream → NNGraphLexer → CommonTokenStream
    │
    ▼  NNGraphParser.program() → parse tree
    │
    ▼  ParseTreeWalker + NNGraphCustomListener
    │    semantic analysis, graph model construction
    │
    ▼  CodeGenerator.generate()
    │    topological sort, variable assignment, PyTorch emit
    │
    ▼  .py output file
```

Entry point: `compile_nng()` in `main.py` (73 lines).

### 4.2 Semantic analyzer (`custom_listener.py`, 293 lines)

Class: `NNGraphCustomListener(NNGraphListener)`

Internal state built during tree walk:
- `self.model_name` — string
- `self.input_id`, `self.input_shape` — string, tuple
- `self.output_id` — string
- `self.nodes` — `dict[str, {"type": str, "params": dict, "line": int}]`
- `self.edges` — `list[{"src": str, "dst": str, "label": str|None, "line": int}]`
- `self.config` — `dict[str, Any]`

Key `exit*` methods:
- `exitShape_expr` — builds `tuple[int]` from `INT_LITERAL` tokens
- `exitValue` — attaches `.pvalue` and `.ptype` to ctx (propagated up tree)
- `exitNode_decl` — validates layer type against `LAYER_SCHEMA`, checks param types
- `exitEdge_decl` — validates src/dst exist in `self.nodes`
- `exitGraph_block` — runs 5 post-graph validation passes

Error format (mirrors EVM project):
```
[line L:C] SemanticError: <message>
```
Warnings use `warn()` and print to stderr without aborting.

**8 validation rules** (in execution order):
1. Unique node IDs — enforced in `exitNode_decl`
2. Undefined edge references — enforced in `exitEdge_decl`
3. Orphan nodes — warning in `_validate_orphans`
4. Residual arity (must be exactly 2) — error in `_validate_residual_arity`
5. Cycle detection via Kahn's algorithm — error in `_validate_no_cycles`
6. Input reachability via BFS — warning in `_validate_reachability`
7. Output reachability — error in `_validate_reachability`
8. Parameter type checking via `LAYER_SCHEMA` — error in `exitNode_decl`

### 4.3 Code generator (`code_generator.py`, 239 lines)

Class: `CodeGenerator`

Public interface:
- `load_from_listener(listener)` — copies graph model from listener
- `generate()` — returns Python source string

Internal methods (in call order):
- `_compute_graph()` — builds `in_edges`, `out_edges` dicts; runs Kahn's sort → `topo_order`
- `_assign_vars()` — assigns Python variable name to each node's output tensor
- `_emit_init()` — one `self.x = nn.X(...)` per node with a module
- `_emit_forward()` — one line per node in topological order
- `_emit_node(node_id)` — dispatches to the correct emit pattern
- `_build_params(layer_type, params, param_name_map)` — maps DSL param names to PyTorch names
- `_assemble(init_lines, forward_lines)` — joins into complete Python source

### 4.4 Variable assignment algorithm

The key invariant: `var_at[n] = 'x'` only when `out_degree[predecessor] == 1`.

Full rules (in `_assign_vars`):
1. Input node: `var_at[input_id] = 'x'`
2. Merge node (in_degree > 1): `var_at[n] = 'x'` if `out_degree[n] <= 1`, else `var_at[n] = node_id`
3. Branch start (predecessor has out_degree > 1): `var_at[n] = node_id`
4. Linear continuation (predecessor has out_degree == 1): `var_at[n] = var_at[predecessor]`

This ensures the input tensor is never overwritten while another branch still holds
a reference. Verified at runtime: `x.data_ptr()` does not change across branch
computations (see traced forward pass output in project notes).

### 4.5 Special code-generation cases

| Pattern | Condition | Emitted code |
|---|---|---|
| Self-attention | `MultiHeadAttn` node | `out, _ = self.attn(x, x, x)` |
| Recurrent | `LSTM` or `GRU` node | `out, _ = self.lstm(x)` |
| Explicit residual | `Residual()` node, 2 in-edges | `x = main_var + shortcut_var` |
| Concat | `Concat(dim=d)` node | `x = torch.cat([a, b, ...], dim=d)` |
| Element-wise add | `Add()` node | `x = a + b` |
| Split | `Split(chunks=n, dim=d)` | `parts = torch.chunk(x, n, dim=d)` |
| Implicit residual | Node with in_degree > 1, `residual_*` labeled in-edge | `x = self.norm(skip + main)` |
| MultiHeadAttn extra arg | Always | adds `batch_first=True` |

### 4.6 PyTorch parameter name mapping

DSL uses shorter names for some parameters. The mapping is in `PYTORCH_MAP`:

| DSL param | PyTorch param |
|---|---|
| `in_ch` | `in_channels` |
| `out_ch` | `out_channels` |
| `kernel` | `kernel_size` |

All other parameter names pass through unchanged.

---

## 5. Evaluation

### 5.1 Test suite

- Total tests: **55**
- Framework: pytest 8.4.2
- Runtime: ~1.09 seconds on local machine

| File | Tests | What it covers |
|---|---|---|
| `tests/test_parser.py` | 8 | Grammar parses all 4 examples with 0 syntax errors; config, label, shape parsing |
| `tests/test_listener.py` | 14 | 4 valid inputs pass; 8 error conditions each raise `SystemExit` at correct line |
| `tests/test_codegen.py` | 19 | `nn` class names, parameter values, forward-pass patterns in generated source |
| `tests/test_e2e.py` | 14 | Full compile + instantiate + forward pass with correct output tensor shape |

### 5.2 Worked examples

All four examples from the language specification compile and run correctly.

**Example 1 — MLP (3-layer classifier):**
- Input: `tensor(784)` → Output: `(batch, 10)`
- Topology: linear chain
- Variable strategy: all `x`
- Generated `forward` matches spec exactly

**Example 2 — ResBlock (convolutional residual block):**
- Input: `tensor(64, 32, 32)` → Output: `(batch, 65536)` after Flatten
- Topology: fan-out from input (out_degree=2), merge at `Residual()`
- Variable strategy: main path uses `conv1`; shortcut preserves original `x`
- Residual add: `x = conv1 + x`
- Spec uses `identity = x` alias; output is mathematically identical

**Example 3 — InceptionBlock (parallel conv branches):**
- Input: `tensor(32, 56, 56)` → Output: `(batch, 48, 56, 56)`
- Topology: input fans out to two branches; merge at `Concat(dim=1)`
- Variable strategy: branch A uses `convA1`; branch B uses `convB1`
- Concat: `x = torch.cat([convA1, convB1], dim=1)`
- Spec uses single-letter names `a`, `b`; output is mathematically identical

**Example 4 — TransformerEncoder (single encoder layer):**
- Input: `tensor(512, 128)` → Output: `(1024, 128)` after `Flatten(start_dim=0, end_dim=1)`
- Topology: two implicit residual sub-layers
  - Sub-layer 1: `x` fans out to `attn` (branch) and `norm1` (skip via `residual_1`)
  - Sub-layer 2: `norm1` fans out to `ff1` (branch) and `norm2` (skip via `residual_2`)
- Variable strategy:
  - `attn` branch: variable `attn`
  - `norm1` node (out_degree=2): variable `norm1`
  - `ff1` branch: variable `ff1`
  - `norm2` node: back to `x`
- Generated forward:
  ```python
  attn, _ = self.attn(x, x, x)
  attn = self.drop1(attn)
  norm1 = self.norm1(x + attn)
  ff1 = self.ff1(norm1)
  ff1 = self.gelu1(ff1)
  ff1 = self.drop2(ff1)
  ff1 = self.ff2(ff1)
  x = self.norm2(norm1 + ff1)
  x = self.out(x)
  return x
  ```

### 5.3 Variable correctness proof

For each branching example, the input tensor's `data_ptr` was traced through
the forward pass:

- **ResBlock**: `x.data_ptr()` equals entry value at the point `x = conv1 + x`
- **Inception**: `x.data_ptr()` equals entry value when both `self.convA1(x)` and `self.convB1(x)` are called
- **Transformer**: `x.data_ptr()` equals entry value when `self.norm1(x + attn)` is called; `norm1.data_ptr()` unchanged when `self.norm2(norm1 + ff1)` is called

---

## 6. Design decisions worth discussing

**Listener over visitor:**
The listener pattern is stateful by design; `exit*` methods accumulate into
`self.nodes` and `self.edges` naturally. A visitor would require explicit return
values through the tree, which adds friction for this use case.

**`CodeGenerator.load_from_listener()` instead of `generate(traversal)`:**
The EVM project in `PycharmProjects/Antlr` passes a flat post-order AST traversal
to `CodeGenerator.generate(traversal)` because it generates stack-based bytecode.
For a graph compiler, the natural input is the graph structure (nodes + edges), not
a linear token stream. Passing the listener directly is cleaner.

**Variable naming `node_id` over single letters:**
The spec examples use `a`, `b`, `identity`, `attn_out`, `ff_out`. This compiler
uses the actual node ID (`convA1`, `conv1`, `attn`, `ff1`) because it's
deterministic, readable, and avoids a counter that would break on larger graphs.
Output is mathematically identical.

**`_FakeCtx` for post-validation errors:**
Errors in `exitGraph_block` occur after the full parse tree walk, so there is no
live `ctx.start.line`. A minimal `_FakeCtx` object carries the line number from
the node's declaration, keeping the error format consistent.

---

## 7. Codebase statistics

| Artifact | Lines |
|---|---|
| `NNGraph.g4` | 56 |
| `custom_listener.py` | 293 |
| `code_generator.py` | 239 |
| `main.py` | 73 |
| `tests/` (4 files) | 575 |
| `required_code_collection/` | ~100 |
| **Total** | ~1336 |

Language: Python 3.14.3
ANTLR version: 4.13.2
PyTorch version: 2.10.0
Test runner: pytest 8.4.2

---

## 8. Future extensions (from specification)

Listed in spec but not implemented:

- Shape inference at compile time
- Quantization metadata annotations
- ONNX export
- Training scaffold (optimizer + loss from `config` block)
- Graphviz visualization
- Macro / template sub-graphs (named reusable blocks)
- Dynamic control flow (`if`/`else` branches)
