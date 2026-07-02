"""
Shape inference pass — propagate tensor shapes through the DAG and catch
dimension mismatches at compile time.
Usage:
    shapes = infer_shapes(nodes, edges, input_id, input_shape, topo_order)
    # shapes: dict  node_id -> tuple  (shape WITHOUT batch dim)
"""
import sys
from collections import deque
def _error(msg, node_id, line):
    print(f"[line {line}] ShapeError at '{node_id}': {msg}", file=sys.stderr)
    raise SystemExit(1)
def _get(params, key, default=None):
    entry = params.get(key)
    if entry is None:
        return default
    return entry["value"]
def _conv_out(in_size, kernel, stride, padding):
    return (in_size + 2 * padding - kernel) // stride + 1
def _infer_node(node_id, info, in_shapes, edges_in):
    """Return output shape for a single node given its incoming shapes."""
    layer = info["type"]
    params = info["params"]
    line = info["line"]
    if layer == "__input__":
        return None
    if not in_shapes:
        _error("No incoming shape available.", node_id, line)
    shape = in_shapes[0]
    if layer in ("ReLU", "Sigmoid", "Tanh", "GELU", "LeakyReLU", "ELU", "Dropout"):
        return shape
    if layer == "Linear":
        in_f = _get(params, "in_features")
        out_f = _get(params, "out_features")
        if in_f is not None and shape[-1] != in_f:
            _error(f"in_features={in_f} but incoming last dim={shape[-1]}.", node_id, line)
        return shape[:-1] + (out_f,)
    if layer == "Conv2d":
        if len(shape) != 3:
            _error(f"Conv2d expects 3D input (C,H,W), got {len(shape)}D.", node_id, line)
        in_ch = _get(params, "in_ch")
        out_ch = _get(params, "out_ch")
        kernel = _get(params, "kernel", 3)
        stride = _get(params, "stride", 1)
        padding = _get(params, "padding", 0)
        if in_ch is not None and shape[0] != in_ch:
            _error(f"in_ch={in_ch} but incoming channels={shape[0]}.", node_id, line)
        h_out = _conv_out(shape[1], kernel, stride, padding)
        w_out = _conv_out(shape[2], kernel, stride, padding)
        return (out_ch, h_out, w_out)
    if layer == "Conv1d":
        if len(shape) != 2:
            _error(f"Conv1d expects 2D input (C,L), got {len(shape)}D.", node_id, line)
        in_ch = _get(params, "in_ch")
        out_ch = _get(params, "out_ch")
        kernel = _get(params, "kernel", 3)
        stride = _get(params, "stride", 1)
        if in_ch is not None and shape[0] != in_ch:
            _error(f"in_ch={in_ch} but incoming channels={shape[0]}.", node_id, line)
        l_out = (shape[1] - kernel) // stride + 1
        return (out_ch, l_out)
    if layer == "BatchNorm2d":
        nf = _get(params, "num_features")
        if nf is not None and len(shape) >= 1 and shape[0] != nf:
            _error(f"num_features={nf} but incoming channels={shape[0]}.", node_id, line)
        return shape
    if layer == "LayerNorm":
        return shape
    if layer in ("MaxPool2d", "AvgPool2d"):
        if len(shape) != 3:
            _error(f"{layer} expects 3D input (C,H,W), got {len(shape)}D.", node_id, line)
        kernel = _get(params, "kernel", 2)
        stride = _get(params, "stride", kernel)
        h_out = (shape[1] - kernel) // stride + 1
        w_out = (shape[2] - kernel) // stride + 1
        return (shape[0], h_out, w_out)
    if layer == "Flatten":
        start = _get(params, "start_dim", 1) - 1
        end = _get(params, "end_dim", -1)
        if end == -1 or end >= len(shape):
            end = len(shape) - 1
        else:
            end = end - 1
        if start < 0:
            start = 0
        flat_size = 1
        for i in range(start, end + 1):
            flat_size *= shape[i]
        return shape[:start] + (flat_size,) + shape[end + 1:]
    if layer == "Embedding":
        edim = _get(params, "embedding_dim")
        return shape + (edim,)
    if layer == "MultiHeadAttn":
        return shape
    if layer in ("LSTM", "GRU"):
        hidden = _get(params, "hidden_size")
        return shape[:-1] + (hidden,)
    if layer == "Softmax":
        return shape
    if layer in ("Add", "Residual"):
        for s in in_shapes[1:]:
            if s != in_shapes[0]:
                _error(f"Shape mismatch in {layer}: {in_shapes[0]} vs {s}.", node_id, line)
        return in_shapes[0]
    if layer == "Concat":
        dim = _get(params, "dim", 1) - 1
        base = list(in_shapes[0])
        for s in in_shapes[1:]:
            for i, (a, b) in enumerate(zip(base, s)):
                if i == dim:
                    continue
                if a != b:
                    _error(f"Concat dim mismatch at axis {i+1}: {a} vs {b}.", node_id, line)
            base[dim] += s[dim]
        return tuple(base)
    if layer == "Split":
        chunks = _get(params, "chunks", 2)
        dim = _get(params, "dim", 1) - 1
        if shape[dim] % chunks != 0:
            _error(f"Cannot split dim {dim+1} (size {shape[dim]}) into {chunks} chunks.", node_id, line)
        new = list(shape)
        new[dim] = shape[dim] // chunks
        return tuple(new)
    _error(f"Unknown layer type '{layer}' for shape inference.", node_id, line)
def infer_shapes(nodes, edges, input_id, input_shape, topo_order):
    """
    Propagate shapes through the graph. Returns dict of node_id -> shape (no batch dim).
    Raises SystemExit on mismatch.
    """
    in_edges = {n: [] for n in nodes}
    for e in edges:
        in_edges[e["dst"]].append(e)
    shapes = {input_id: input_shape}
    for node_id in topo_order:
        if node_id == input_id:
            continue
        info = nodes[node_id]
        my_in = in_edges[node_id]
        in_shapes = [shapes[e["src"]] for e in my_in]
        out_shape = _infer_node(node_id, info, in_shapes, my_in)
        shapes[node_id] = out_shape
    return shapes
