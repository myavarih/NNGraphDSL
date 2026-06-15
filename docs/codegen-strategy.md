# Code generation strategy

## Overview

`CodeGenerator.generate()` (`code_generator.py:54`) runs four passes in fixed order:

1. `_compute_graph()` — build adjacency lists; produce topological order
2. `_assign_vars()` — assign a Python variable name to each node's output tensor
3. `_emit_init()` + `_emit_forward()` — produce line lists
4. `_assemble()` — join into final Python source

---

## Pass 1: topological sort

`_compute_graph()` builds two dicts:

- `self.in_edges[node_id]` — list of edges arriving at `node_id`
- `self.out_edges[node_id]` — list of edges leaving `node_id`

It then runs Kahn's algorithm (BFS on nodes with `in_degree == 0`) to produce
`self.topo_order`. The semantic analyser already verified the graph is a DAG, so
Kahn's always completes successfully here.

The input pseudo-node (`__input__`) appears first in `topo_order` because it has
no incoming edges.

---

## Pass 2: variable assignment

This is the most important pass. It answers: what Python variable holds each node's
output tensor?

For a linear chain every node can reuse `x`. For a branching graph, reusing `x`
at a branch start would overwrite the shared tensor before other branches read it.

### The invariant

`var_at[n] = 'x'` only when `out_degree[predecessor] == 1`.

When a node's predecessor fans out to multiple destinations (`out_degree > 1`),
that predecessor's tensor is still needed by other branches. Assigning `x` at this
point would overwrite it. Instead, the node gets a variable named after its own ID.

### Four rules (in `_assign_vars`, code_generator.py:83)

```
1. input node:
       var_at[input_id] = 'x'

2. merge node (in_degree > 1):
       var_at[n] = 'x'         if out_degree[n] <= 1
       var_at[n] = node_id     if out_degree[n] > 1

3. branch start (predecessor has out_degree > 1):
       var_at[n] = node_id

4. linear continuation (predecessor has out_degree == 1):
       var_at[n] = var_at[predecessor]
```

### Traced examples

**MLP (linear chain):**

```
x → fc1 → relu1 → fc2 → relu2 → drop → fc3 → out
```

Every predecessor has `out_degree == 1`. Rule 4 applies everywhere.
All nodes inherit `var_at = 'x'`. Generated `forward` uses `x` on every line.

**ResBlock (fan-out then merge):**

```
x ──→ conv1 ──→ bn1 ──→ relu1 ──→ conv2 ──→ bn2 ──→ skip (Residual)
│                                                           │
└───────────────────────────────────────────────────────────┘
```

`x` fans out to `conv1` and `skip` (shortcut edge). `out_degree[x] = 2`.

- `conv1`: rule 3 → `var_at = 'conv1'`
- `bn1`, `relu1`, `conv2`, `bn2`: rule 4, inherit → all `'conv1'`
- `skip` (Residual, in_degree=2): rule 2, `out_degree[skip]=1` → `var_at = 'x'`
  - The shortcut edge carries the original `x`; the main edge carries `conv1`
  - Generated: `x = conv1 + x`
- `flatten`, `out`: rule 4 → `'x'`

**InceptionBlock (two parallel branches):**

```
x ──→ convA1 ──→ convA2 ──→
│                            cat1 (Concat) ──→ bn ──→ out
└──→ convB1 ──→ convB2 ──→
```

`out_degree[x] = 2`.

- `convA1`, `convB1`: rule 3 → `var_at = 'convA1'`, `'convB1'`
- `convA2`: rule 4, inherits `'convA1'`
- `convB2`: rule 4, inherits `'convB1'`
- `cat1` (Concat, in_degree=2): rule 2, `out_degree=1` → `var_at = 'x'`
  - Generated: `x = torch.cat([convA1, convB1], dim=1)`
- `bn`, `out`: rule 4 → `'x'`

**TransformerEncoder (two implicit residual sub-layers):**

```
x ──→ attn ──→ drop1 ──→
│                         norm1 ──→ ff1 ──→ gelu1 ──→ drop2 ──→ ff2 ──→
│                [residual_1]                                              norm2 ──→ out ──→ flatten
└────────────────────────────────────────────────────────────────────────[residual_2]
```

`out_degree[x] = 2` (goes to `attn` and `norm1`).
`out_degree[norm1] = 2` (goes to `ff1` and `norm2`).

- `attn`: rule 3 → `'attn'`
- `drop1`: rule 4, inherits `'attn'`
- `norm1` (in_degree=2, out_degree=2): rule 2 → `'norm1'`
- `ff1`: rule 3 (predecessor `norm1` has `out_degree=2`) → `'ff1'`
- `gelu1`, `drop2`, `ff2`: rule 4, inherit `'ff1'`
- `norm2` (in_degree=2, out_degree=1): rule 2 → `'x'`
- `out`, `flatten`: rule 4 → `'x'`

Generated forward lines for the two norm nodes:
```python
norm1 = self.norm1(x + attn)
...
x = self.norm2(norm1 + ff1)
```

### Correctness

The invariant guarantees `x` is never overwritten while another branch holds a live
reference. Verified at runtime: `x.data_ptr()` does not change between entering
`self.convA1(x)` and `self.convB1(x)` in the Inception example, and between
entering `self.attn(x, x, x)` and `self.norm1(x + attn)` in the Transformer.

---

## Pass 3: emission

### `_emit_init()` (code_generator.py:115)

Iterates `topo_order`. For each node not in `NO_MODULE_NODES` (`Add`, `Concat`,
`Residual`, `Split`, `__input__`), emits one line:

```python
self.<id> = nn.X(<pytorch_params>)
```

`_build_params()` maps DSL parameter names to PyTorch names using `PYTORCH_MAP`:

| DSL | PyTorch |
|---|---|
| `in_ch` | `in_channels` |
| `out_ch` | `out_channels` |
| `kernel` | `kernel_size` |

All other names pass through unchanged.

`MultiHeadAttn` always appends `batch_first=True` regardless of what the user wrote.

### `_emit_node()` dispatch (code_generator.py:149)

Called per node in topological order. Dispatch order:

1. `__input__` → return `[]` (no line)
2. `Residual` → split in-edges by label; emit `out = main + skip`
3. `Add` → emit `out = a + b + ...`
4. `Concat` → emit `out = torch.cat([a, b, ...], dim=d)`
5. `Split` → emit `parts = torch.chunk(x, n, dim=d)`
6. node with `in_degree > 1` and a `residual_*` labeled in-edge → emit `out = self.id(skip + main)`
7. `MultiHeadAttn` → emit `out, _ = self.id(x, x, x)`
8. `LSTM` or `GRU` → emit `out, _ = self.id(x)`
9. default → emit `out = self.id(x)`

---

## Pass 4: assembly

`_assemble()` (code_generator.py:210) joins the pieces in this order:

```python
import torch
import torch.nn as nn


class <ModelName>(nn.Module):
    def __init__(self):
        super(<ModelName>, self).__init__()
        <init_lines>

    def forward(self, x):
        <forward_lines>
        return <output_var>


if __name__ == '__main__':
    device = torch.device('<device>')
    model = <ModelName>().to(device)
    x = torch.randn(<batch_size>, <input_shape...>).to(device)
    print(model(x).shape)
```

`batch_size` defaults to `1` and `device` to `'cpu'` if the `config` block is absent.
