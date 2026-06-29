"""
Graphviz DOT visualization of the NNGraph model graph.

Usage:
    dot_source = generate_dot(nodes, edges, input_id, output_id, model_name, shapes=None)
"""


NODE_COLORS = {
    "__input__":     "#4CAF50",
    "Linear":        "#2196F3",
    "Conv2d":        "#FF9800",
    "Conv1d":        "#FF9800",
    "BatchNorm2d":   "#9C27B0",
    "LayerNorm":     "#9C27B0",
    "MaxPool2d":     "#00BCD4",
    "AvgPool2d":     "#00BCD4",
    "Dropout":       "#795548",
    "Flatten":       "#607D8B",
    "Embedding":     "#E91E63",
    "MultiHeadAttn": "#F44336",
    "LSTM":          "#3F51B5",
    "GRU":           "#3F51B5",
    "ReLU":          "#8BC34A",
    "Sigmoid":       "#8BC34A",
    "Tanh":          "#8BC34A",
    "GELU":          "#8BC34A",
    "LeakyReLU":     "#8BC34A",
    "ELU":           "#8BC34A",
    "Softmax":       "#CDDC39",
    "Add":           "#FF5722",
    "Concat":        "#FF5722",
    "Residual":      "#FF5722",
    "Split":         "#FF5722",
}

DEFAULT_COLOR = "#BDBDBD"


def _escape(s):
    return s.replace('"', '\\"').replace('\n', '\\n')


def _node_label(node_id, info, shapes):
    layer = info["type"]
    if layer == "__input__":
        label = f"{node_id}\\n(input)"
    else:
        params_parts = []
        for k, v in info["params"].items():
            params_parts.append(f"{k}={v['value']}")
        params_str = ", ".join(params_parts)
        label = f"{node_id}\\n{layer}"
        if params_str:
            label += f"\\n{params_str}"

    if shapes and node_id in shapes and shapes[node_id] is not None:
        label += f"\\nshape: {shapes[node_id]}"

    return label


def generate_dot(nodes, edges, input_id, output_id, model_name, shapes=None):
    lines = [
        f'digraph "{_escape(model_name)}" {{',
        '    rankdir=TB;',
        '    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10];',
        '    edge [fontname="Helvetica", fontsize=8];',
        '',
    ]

    for node_id, info in nodes.items():
        color = NODE_COLORS.get(info["type"], DEFAULT_COLOR)
        label = _node_label(node_id, info, shapes)
        peripheries = 2 if node_id == output_id else 1
        lines.append(f'    "{_escape(node_id)}" [label="{label}", fillcolor="{color}", '
                     f'fontcolor="white", peripheries={peripheries}];')

    lines.append('')

    for e in edges:
        attrs = []
        if e.get("label"):
            attrs.append(f'label="{_escape(e["label"])}"')
            attrs.append('style=dashed')
        attr_str = f' [{", ".join(attrs)}]' if attrs else ''
        lines.append(f'    "{_escape(e["src"])}" -> "{_escape(e["dst"])}"{attr_str};')

    lines.append('}')
    return '\n'.join(lines) + '\n'
