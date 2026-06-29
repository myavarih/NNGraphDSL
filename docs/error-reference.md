# Error reference

## Error format

Every fatal error prints to stderr and exits with code 1:

```
[line L:C] SemanticError: <message>
```

`L` is the 1-based line number. `C` is the 0-based column of the token that triggered
the error. Post-graph errors (raised inside `exitGraph_block` validation passes) use
the line of the node's original declaration.

Warnings print to stderr and do not abort:

```
[line L] Warning: <message>
```

---

## Syntax errors

Produced by ANTLR4 before the semantic analyser runs. The compiler prints:

```
[ERROR] N syntax error(s) in '<path>'. Aborting.
```

and exits with code 1. Fix the grammar of the `.nng` file. Common causes:
- Missing closing brace
- Misspelled keyword (e.g. `Mode` instead of `model`)
- Invalid shape syntax (e.g. `tensor 784` instead of `tensor(784)`)
- Arrow written as `->` with a space before `>` — the lexer matches `ARROW` as `->`,
  so `- >` is two tokens and fails to parse

---

## Semantic errors

### Duplicate node ID

```
[line L:C] SemanticError: Duplicate node ID '<id>'.
```

Two `node` declarations in the same `graph` block use the same identifier. Node IDs
must be unique within a graph. Raised in `exitNode_decl` (`custom_listener.py:125`).

### Unknown layer type

```
[line L:C] SemanticError: Unknown layer type '<name>'.
```

The identifier after `:` in a `node` declaration is not in the supported layer set.
Raised in `exitNode_decl` (`custom_listener.py:128`). Check spelling. Supported
types are listed in `custom_listener.py:LAYER_SCHEMA`.

### Unknown parameter

```
[line L:C] SemanticError: Unknown parameter '<param>' for layer '<layer>'.
```

A parameter name does not appear in `LAYER_SCHEMA` for the given layer type. Raised
in `exitNode_decl` (`custom_listener.py:133`).

### Parameter type mismatch

```
[line L:C] SemanticError: Parameter '<param>' of '<layer>' expects <type>, got <actual>.
```

The value provided for a parameter has the wrong type. Common cases:

| Error | Fix |
|---|---|
| `Dropout(p=1)` — expects `float`, got `int` | Write `p=1.0` |
| `Linear(in_features=1.0, ...)` — expects `int`, got `float` | Write `in_features=1` |

Raised in `exitNode_decl` (`custom_listener.py:136`).

### Edge references undefined node

```
[line L:C] SemanticError: Edge references undefined node '<id>'.
  Hint: Declared nodes are: [<list>]
```

An `edge` declaration names a source or destination that has not been declared with
`node`. The input node (from `input_decl`) counts as declared. Raised in
`exitEdge_decl` (`custom_listener.py:159`).

### Output node not declared

```
[line L:C] SemanticError: Output node '<id>' is not declared in the graph block.
```

The identifier in `output <id>` inside the `model` block does not match any `node`
declaration. Raised in `_validate_output_declared` (`custom_listener.py:201`).

### Residual arity error

```
[line L:0] SemanticError: Node '<id>' (Residual) has N incoming edge(s); expected exactly 2.
```

A `Residual()` node has anything other than exactly 2 incoming edges. Raised in
`_validate_residual_arity` (`custom_listener.py:237`). Add or remove edges so the
node has exactly one main-path edge and one shortcut edge.

### Cycle detected

```
[line L:C] SemanticError: Cycle detected involving nodes: [<list>].
  Hint: NNGraph graphs must be acyclic DAGs.
```

Kahn's algorithm visited fewer nodes than the graph contains, meaning at least one
cycle exists. The listed nodes are those whose `in_degree` never reached zero during
Kahn's BFS. Raised in `_validate_no_cycles` (`custom_listener.py:261`).

### Output unreachable

```
[line L:C] SemanticError: Output node '<output_id>' is not reachable from input '<input_id>'.
```

BFS from the input node did not reach the declared output. The output node either
has no path from the input, or the graph is disconnected. Raised in
`_validate_reachability` (`custom_listener.py:290`).

### Orphan node

```
[line L:0] SemanticError: Node '<id>' is declared but never referenced in any edge.
```

A node appears in the `graph` block but is mentioned in no `edge` declaration. This
almost always indicates a wiring mistake (misspelled node name in an edge, or a
forgotten edge). Raised in `_validate_orphans` (`custom_listener.py:221`).

### Node unreachable from input

```
[line L:0] SemanticError: Node '<id>' is not reachable from input '<input_id>'.
```

BFS from the input node did not reach this node. The node is part of an isolated
subgraph and would never participate in the forward pass. Raised in
`_validate_reachability` (`custom_listener.py:287`).

---

## Warnings

Warnings print to stderr but do not stop compilation.

Currently no conditions produce warnings. Orphan nodes and unreachable nodes, which
were previously warnings, are now errors (see above).

---

## Shape inference errors

Produced by the shape inference pass after semantic analysis. Format:

```
[line L] ShapeError at '<node_id>': <message>
```

### Dimension mismatch

```
[line L] ShapeError at 'fc2': in_features=256 but incoming last dim=128.
```

The `in_features` or `in_ch` parameter does not match the actual shape flowing into
the node. Fix the parameter value or the upstream architecture.

### Wrong input rank

```
[line L] ShapeError at 'conv1': Conv2d expects 3D input (C,H,W), got 1D.
```

The layer requires a specific number of dimensions. Check the input shape and any
preceding `Flatten` or `Reshape` operations.

### Split not divisible

```
[line L] ShapeError at 'split': Cannot split dim 1 (size 7) into 2 chunks.
```

The tensor dimension is not evenly divisible by the chunk count.

### Concat dimension mismatch

```
[line L] ShapeError at 'cat1': Concat dim mismatch at axis 2: 56 vs 28.
```

Incoming tensors to `Concat` have mismatched sizes on non-concat dimensions.

---

## Quantization errors

### Unknown quant mode

```
[line L:C] SemanticError: Unknown quant mode 'int4'. Valid: ['dynamic', 'qat', 'static'].
```

The `quant` parameter value is not one of the supported quantization modes.

---

## Runtime errors (generated code)

These are not compiler errors. They surface when the generated `.py` file is run:

| Symptom | Likely cause |
|---|---|
| Shape mismatch in `forward` | `in_features`/`in_ch` does not match actual tensor dimension |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | `Linear.in_features` wrong |
| `RuntimeError: Given groups=1, weight...` | `Conv2d.in_ch` does not match actual channel count |
| CUDA out of memory | `batch_size` in `config` too large for available GPU memory |

The shape inference pass now catches most dimension mismatches at compile time.
Remaining runtime errors may surface for edge cases not covered by static analysis.
