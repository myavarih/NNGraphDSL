# Language reference

## Program structure

A `.nng` file contains three top-level blocks. The `config` block is optional:

```
model <Name> {
    input  <id> : tensor(<shape>)
    output <id>
}

graph {
    node <id> : <LayerType>(<params>)
    ...
    edge <src> -> <dst>
    edge <src> -> <dst> [label="<string>"]
    ...
}

config {
    <key> = <value>
    ...
}
```

The `model` block gives the class its name and declares the input and output nodes.
The `graph` block declares all layers and wires them together with directed edges.
The `config` block sets values used in the generated `__main__` block.

---

## Identifiers

Node IDs and the model name follow the same rules as Python identifiers: letters,
digits, and underscores, not starting with a digit. The input node's ID is also used
as the first argument to `forward(self, x)` — the generated code always names it `x`
regardless of what the `.nng` file calls it.

---

## Shapes

A shape is a parenthesised tuple of positive integers. Use one integer for 1D inputs
and comma-separated integers for higher dimensions:

```
input x : tensor(784)          // vector of length 784
input x : tensor(64, 32, 32)   // 64 channels, 32x32 spatial
input x : tensor(512, 128)     // 512 tokens, embedding dim 128
```

Shapes appear only in `input_decl`. They are not inferred or checked at compile time;
mismatches surface at PyTorch runtime.

---

## Values

| Type | Syntax | Python type |
|---|---|---|
| integer | `128` | `int` |
| float | `0.1` | `float` |
| boolean | `true` / `false` | `bool` |
| string | `"relu"` | `str` |
| shape | `(64, 32, 32)` | `tuple[int, ...]` |
| null | `None` | `NoneType` |

The distinction between integer and float matters: passing `p=1` to `Dropout`
(which expects `float`) is a type error. Pass `p=1.0` instead.

---

## Layer types

### Parameterized layers

These emit a `self.<id> = nn.X(...)` line in `__init__`.

| DSL keyword | PyTorch class | Parameters |
|---|---|---|
| `Linear` | `nn.Linear` | `in_features: int`, `out_features: int`, `bias: bool` |
| `Conv2d` | `nn.Conv2d` | `in_ch: int`, `out_ch: int`, `kernel: int`, `stride: int`, `padding: int` |
| `Conv1d` | `nn.Conv1d` | `in_ch: int`, `out_ch: int`, `kernel: int`, `stride: int` |
| `BatchNorm2d` | `nn.BatchNorm2d` | `num_features: int` |
| `LayerNorm` | `nn.LayerNorm` | `normalized_shape: int \| tuple` |
| `MaxPool2d` | `nn.MaxPool2d` | `kernel: int`, `stride: int` |
| `AvgPool2d` | `nn.AvgPool2d` | `kernel: int`, `stride: int` |
| `Dropout` | `nn.Dropout` | `p: float` |
| `Flatten` | `nn.Flatten` | `start_dim: int`, `end_dim: int` |
| `Embedding` | `nn.Embedding` | `num_embeddings: int`, `embedding_dim: int` |
| `MultiHeadAttn` | `nn.MultiheadAttention` | `embed_dim: int`, `num_heads: int` |
| `LSTM` | `nn.LSTM` | `input_size: int`, `hidden_size: int`, `num_layers: int` |
| `GRU` | `nn.GRU` | `input_size: int`, `hidden_size: int` |

`Conv2d` and `Conv1d` use shorter DSL names for channel and kernel parameters:
`in_ch` maps to `in_channels`, `out_ch` to `out_channels`, `kernel` to `kernel_size`.

`MultiHeadAttn` always adds `batch_first=True` in the generated constructor call.
The `forward` call uses self-attention: `self.<id>(x, x, x)`.

`LSTM` and `GRU` emit tuple unpacking: `out, _ = self.<id>(x)`.

### Activations

These have no required parameters. Optional parameters are listed where they exist:

| DSL keyword | PyTorch class | Optional parameters |
|---|---|---|
| `ReLU` | `nn.ReLU` | none |
| `Sigmoid` | `nn.Sigmoid` | none |
| `Tanh` | `nn.Tanh` | none |
| `GELU` | `nn.GELU` | none |
| `Softmax` | `nn.Softmax` | `dim: int` |
| `LeakyReLU` | `nn.LeakyReLU` | `negative_slope: float` |
| `ELU` | `nn.ELU` | `alpha: float` |

### Graph operations

These do not emit a `self.<id>` assignment in `__init__`. They emit an expression
in `forward` instead:

| DSL keyword | Parameters | Emitted code |
|---|---|---|
| `Add` | none | `out = a + b` |
| `Concat` | `dim: int` | `out = torch.cat([a, b, ...], dim=d)` |
| `Residual` | none | `out = main + shortcut` |
| `Split` | `chunks: int`, `dim: int` | `parts = torch.chunk(x, n, dim=d)` |

`Residual` requires exactly two incoming edges. One must be labeled `shortcut`.

---

## Edges

A simple edge connects two nodes:

```
edge conv1 -> bn1
```

A labeled edge carries a string annotation:

```
edge x -> skip [label="shortcut"]
edge x -> norm1 [label="residual_1"]
```

Labels have two roles:

**`"shortcut"`** — marks the skip connection into a `Residual()` node. The compiler
separates shortcut edges from main-path edges and emits `main + shortcut`.

**`"residual_*"` (any label starting with `residual`)** — marks the skip connection
into a regular module node that has two incoming edges. This is the transformer
sub-layer pattern. The compiler emits `self.norm(skip + main)` inline, without a
separate `Residual()` node.

Edge order within the `graph` block does not affect topological sort. The compiler
computes order from the graph structure, not from declaration order.

---

## Config block

The `config` block is optional. Keys and their effects on generated code:

| Key | Type | Effect |
|---|---|---|
| `batch_size` | `int` | first dimension of `torch.randn(...)` in `__main__` |
| `device` | `string` | `torch.device(...)` in `__main__` |

If `batch_size` is absent, the generated `__main__` uses `1`. If `device` is absent,
it uses `'cpu'`.

Config values are not used inside `__init__` or `forward`. They affect only the
run-at-the-bottom test block.

---

## Comments

Two comment styles:

```
// single-line comment

/* multi-line
   comment */
```

Both are stripped by the lexer and have no effect on compilation.

---

## Complete example

A minimal `.nng` file illustrating all three blocks:

```
model MLP {
    input  x   : tensor(784)
    output out
}

graph {
    node fc1   : Linear(in_features=784, out_features=256)
    node relu1 : ReLU()
    node fc2   : Linear(in_features=256, out_features=128)
    node relu2 : ReLU()
    node drop  : Dropout(p=0.3)
    node fc3   : Linear(in_features=128, out_features=10)
    node out   : Softmax(dim=1)

    edge x     -> fc1
    edge fc1   -> relu1
    edge relu1 -> fc2
    edge fc2   -> relu2
    edge relu2 -> drop
    edge drop  -> fc3
    edge fc3   -> out
}

config {
    batch_size = 64
    device = "cuda"
}
```
