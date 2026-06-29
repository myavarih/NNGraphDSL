# NNGraph DSL

A small compiler that takes a neural network described as a graph and turns it into
runnable PyTorch code. You write the architecture once in a readable `.nng` file;
the compiler handles the boilerplate.

The design follows the NADER paper, which treats neural architectures as directed
computational graphs. The compiler is built with ANTLR4 and targets PyTorch 2.x.

---

## What it does

You write this:

```
model MLP {
    input x : tensor(784)
    output out
}

graph {
    node fc1   : Linear(in_features=784, out_features=256)
    node relu1 : ReLU()
    node fc2   : Linear(in_features=256, out_features=128)
    node relu2 : ReLU()
    node fc3   : Linear(in_features=128, out_features=10)
    node out   : Softmax(dim=1)

    edge x -> fc1
    edge fc1 -> relu1
    edge relu1 -> fc2
    edge fc2 -> relu2
    edge relu2 -> fc3
    edge fc3 -> out
}
```

The compiler produces this:

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=128, out_features=10)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x
```

It also handles branching topologies: residual connections, parallel branches with
`Concat`, and transformer-style implicit residuals where a `LayerNorm` takes two
in-edges (one labeled `residual_*`).

---

## Requirements

- Python 3.9+
- `antlr4-python3-runtime` 4.13
- Java 11+ (to regenerate the lexer/parser from the grammar; not needed at runtime)
- PyTorch 2.x (to run the generated code)

Install the Python dependencies:

```bash
pip install antlr4-python3-runtime==4.13.*
```

The ANTLR jar is at `/usr/share/java/antlr-4.13.2-complete.jar` on most Linux
systems. If yours is elsewhere, see [regenerating the parser](#regenerating-the-parser).

---

## Usage

Compile a `.nng` file to a `.py` file:

```bash
python main.py -i input/mlp.nng -o output/mlp.py
```

Validate without generating output:

```bash
python main.py -i input/resblock.nng --check-only
```

Print generated code to stdout instead of writing a file:

```bash
python main.py -i input/inception.nng --print
```

Show inferred tensor shapes per node:

```bash
python main.py -i input/mlp.nng --show-shapes
```

Export a Graphviz DOT visualization:

```bash
python main.py -i input/inception.nng --graph-viz output/inception.dot
dot -Tpng output/inception.dot -o output/inception.png
```

Export to ONNX format:

```bash
python main.py -i input/mlp.nng --onnx output/mlp.onnx
```

Run the generated model:

```bash
python output/mlp.py
# → torch.Size([64, 10])
```

---

## Language reference

### Program structure

Every `.nng` file has three sections. The `config` block is optional:

```
model <Name> {
    input <id> : tensor(<shape>)
    output <id>
}

graph {
    node <id> : <LayerType>(<params>)
    ...
    edge <src> -> <dst>
    ...
}

config {
    batch_size = <int>
    device = "cpu" | "cuda"
    loss = "CrossEntropyLoss"      // enables training scaffold
    optimizer = "Adam"             // Adam | SGD | AdamW | RMSprop
    lr = 0.001
    epochs = 10
}
```

### Shapes

Shapes are tuples of integers. A 1D input uses a single integer; 2D and 3D inputs
use comma-separated values:

```
input x : tensor(784)          // 784-dimensional vector
input x : tensor(64, 32, 32)   // 64 channels, 32×32 spatial
input x : tensor(512, 128)     // sequence of 512 tokens, dim 128
```

### Layer types

| DSL keyword | PyTorch class | Parameters |
|---|---|---|
| `Linear` | `nn.Linear` | `in_features`, `out_features`, `bias` |
| `Conv2d` | `nn.Conv2d` | `in_ch`, `out_ch`, `kernel`, `stride`, `padding` |
| `Conv1d` | `nn.Conv1d` | `in_ch`, `out_ch`, `kernel`, `stride` |
| `BatchNorm2d` | `nn.BatchNorm2d` | `num_features` |
| `LayerNorm` | `nn.LayerNorm` | `normalized_shape` |
| `MaxPool2d` | `nn.MaxPool2d` | `kernel`, `stride` |
| `AvgPool2d` | `nn.AvgPool2d` | `kernel`, `stride` |
| `Dropout` | `nn.Dropout` | `p` |
| `Flatten` | `nn.Flatten` | `start_dim`, `end_dim` |
| `Embedding` | `nn.Embedding` | `num_embeddings`, `embedding_dim` |
| `MultiHeadAttn` | `nn.MultiheadAttention` | `embed_dim`, `num_heads` |
| `LSTM` | `nn.LSTM` | `input_size`, `hidden_size`, `num_layers` |
| `GRU` | `nn.GRU` | `input_size`, `hidden_size` |

Activations: `ReLU()`, `Sigmoid()`, `Tanh()`, `GELU()`, `Softmax(dim=1)`,
`LeakyReLU(negative_slope=0.01)`, `ELU(alpha=1.0)`.

Graph operations (no `nn.Module` in `__init__`): `Add()`, `Concat(dim=1)`,
`Residual()`, `Split(chunks=2, dim=1)`.

### Quantization annotations

Any node can carry a `quant` parameter to enable quantization-aware compilation:

```
node fc1 : Linear(in_features=784, out_features=128, quant="dynamic")
node fc2 : Linear(in_features=128, out_features=10, quant="qat")
```

Supported modes: `"dynamic"`, `"static"`, `"qat"`. QAT nodes are wrapped in
`torch.quantization.QuantWrapper`. When any node has a quant annotation, the
generated `forward()` includes `QuantStub`/`DeQuantStub` calls.

### Edges

A simple directed edge:

```
edge conv1 -> bn1
```

A labeled edge — the label is used by the compiler to identify skip connections:

```
edge x -> skip [label="shortcut"]
edge x -> norm1 [label="residual_1"]
```

`Residual()` nodes require exactly two incoming edges. One must be labeled
`shortcut`; the other is the main path. The compiler emits `x = main + shortcut`.

For transformer-style sub-layers, connect the skip directly to the norm node with
a label starting with `residual`. The compiler detects this and emits
`self.norm(skip_var + main_var)` inline.

### Comments

```
// single-line comment

/* multi-line
   comment */
```

---

## Semantic rules

The compiler rejects invalid graphs before emitting any code. Errors print the
source line and column:

```
[line 14:4] SemanticError: Edge references undefined node 'fc99'.
  Hint: Declared nodes are: ['x', 'fc1', 'relu1', 'fc2', 'out']
```

The checks, in order:

1. **Unique node IDs** — duplicate IDs in the `graph` block are rejected.
2. **Undefined references** — both endpoints of every `edge` must be declared nodes.
3. **Orphan nodes** — a node never referenced in any edge is an error.
4. **Residual arity** — `Residual()` nodes with anything other than 2 incoming edges are rejected.
5. **Cycle detection** — the graph must be a DAG. Any cycle is rejected.
6. **Reachability** — nodes unreachable from the input are an error. An unreachable output node is also an error.
7. **Parameter types** — each parameter is checked against the expected type for its layer. Passing `p=1` to `Dropout` (expects `float`) is an error.
8. **Unknown layers** — layer names not in the supported set are rejected immediately.

---

## How the compiler handles branching

The code generator assigns a Python variable to each node's output. For a linear
chain, every node reuses `x`. When a node fans out to multiple destinations, all
downstream branch nodes get a variable named after the first node in their branch,
so the original tensor is not overwritten while other branches still need it.

Merge nodes (`Concat`, `Add`, `Residual`) collect their input variables and return
to `x` afterwards. Implicit residuals (a norm layer with two in-edges, one labeled
`residual_*`) receive the sum inline: `self.norm1(x + attn)`.

The `data_ptr` of the input tensor does not change across branch computations,
which you can verify by running any of the four examples under `input/`.

---

## Examples

Four worked examples are in `input/`. Each matches the corresponding example from
the language specification:

| File | Architecture | Output shape |
|---|---|---|
| `input/mlp.nng` | 3-layer MLP | `(batch, 10)` |
| `input/resblock.nng` | ResNet-style block with skip connection | `(batch, 65536)` after Flatten |
| `input/inception.nng` | Two parallel conv branches merged with Concat | `(batch, 48, 56, 56)` |
| `input/transformer.nng` | Single transformer encoder layer | `(1024, 128)` after Flatten |

Compile and run the ResBlock example:

```bash
python main.py -i input/resblock.nng -o output/resblock.py
python output/resblock.py
# → torch.Size([1, 65536])
```

---

## Running the tests

```bash
python -m pytest tests/ -v
```

55 tests across four files:

- `tests/test_parser.py` — grammar parses all four examples with zero syntax errors.
- `tests/test_listener.py` — valid graphs pass; each of the 8 error conditions raises `SystemExit` at the correct line.
- `tests/test_codegen.py` — generated code contains the expected `nn` classes, parameter names, and forward-pass patterns.
- `tests/test_e2e.py` — each example compiles, instantiates, and produces the correct output tensor shape under PyTorch.

---

## Regenerating the parser

If you edit `NNGraph.g4`, regenerate the lexer and parser with:

```bash
java -jar /usr/share/java/antlr-4.13.2-complete.jar \
    -Dlanguage=Python3 -listener -o gen/ NNGraph.g4
```

The files in `gen/` are generated and should not be edited by hand.

---

## Project layout

```
NNGraph.g4                   ANTLR4 grammar — source of truth for syntax
gen/                         generated lexer and parser (do not edit)
required_code_collection/    AST node class and tree utilities
custom_listener.py           semantic analysis — exit* methods, error reporting
code_generator.py            topological sort, variable assignment, PyTorch emit
shape_inference.py           compile-time tensor shape propagation
graph_viz.py                 Graphviz DOT visualization generator
onnx_export.py               ONNX export via torch.onnx.export
main.py                      CLI: compile_nng() + argparse
input/                       .nng source files
output/                      generated .py files
tests/                       pytest suite
```

---

## Planned extensions

These are listed in the specification but not yet implemented:

- Macros / templates — reusable named sub-graphs (e.g., a named ResBlock that can be instantiated multiple times).
- Dynamic control flow — conditional branches expressed with an `if`/`else`-style construct.
