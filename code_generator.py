from collections import deque

NO_MODULE_NODES = {"Add", "Concat", "Residual", "Split", "__input__"}

PYTORCH_MAP = {
    "Linear":        ("nn.Linear",            {"in_features": "in_features", "out_features": "out_features", "bias": "bias"}),
    "Conv2d":        ("nn.Conv2d",             {"in_ch": "in_channels", "out_ch": "out_channels", "kernel": "kernel_size", "stride": "stride", "padding": "padding"}),
    "Conv1d":        ("nn.Conv1d",             {"in_ch": "in_channels", "out_ch": "out_channels", "kernel": "kernel_size", "stride": "stride"}),
    "BatchNorm2d":   ("nn.BatchNorm2d",        {"num_features": "num_features"}),
    "LayerNorm":     ("nn.LayerNorm",          {"normalized_shape": "normalized_shape"}),
    "MaxPool2d":     ("nn.MaxPool2d",          {"kernel": "kernel_size", "stride": "stride"}),
    "AvgPool2d":     ("nn.AvgPool2d",          {"kernel": "kernel_size", "stride": "stride"}),
    "Dropout":       ("nn.Dropout",            {"p": "p"}),
    "Flatten":       ("nn.Flatten",            {"start_dim": "start_dim", "end_dim": "end_dim"}),
    "Embedding":     ("nn.Embedding",          {"num_embeddings": "num_embeddings", "embedding_dim": "embedding_dim"}),
    "MultiHeadAttn": ("nn.MultiheadAttention", {"embed_dim": "embed_dim", "num_heads": "num_heads"}),
    "LSTM":          ("nn.LSTM",               {"input_size": "input_size", "hidden_size": "hidden_size", "num_layers": "num_layers"}),
    "GRU":           ("nn.GRU",                {"input_size": "input_size", "hidden_size": "hidden_size"}),
    "ReLU":          ("nn.ReLU",               {}),
    "Sigmoid":       ("nn.Sigmoid",            {}),
    "Tanh":          ("nn.Tanh",               {}),
    "GELU":          ("nn.GELU",               {}),
    "Softmax":       ("nn.Softmax",            {"dim": "dim"}),
    "LeakyReLU":     ("nn.LeakyReLU",          {"negative_slope": "negative_slope"}),
    "ELU":           ("nn.ELU",                {"alpha": "alpha"}),
}


class CodeGenerator:

    def __init__(self):
        self.model_name  = None
        self.input_id    = None
        self.input_shape = None
        self.output_id   = None
        self.nodes       = {}
        self.edges       = []
        self.config      = {}

        self.topo_order = []
        self.in_edges   = {}
        self.out_edges  = {}
        self.var_at     = {}

    def load_from_listener(self, listener):
        self.model_name  = listener.model_name
        self.input_id    = listener.input_id
        self.input_shape = listener.input_shape
        self.output_id   = listener.output_id
        self.nodes       = listener.nodes
        self.edges       = listener.edges
        self.config      = listener.config

    def generate(self):
        self._compute_graph()
        self._assign_vars()
        init_lines    = self._emit_init()
        forward_lines = self._emit_forward()
        return self._assemble(init_lines, forward_lines)

    # ── graph utilities ────────────────────────────────────────

    def _compute_graph(self):
        self.in_edges  = {n: [] for n in self.nodes}
        self.out_edges = {n: [] for n in self.nodes}
        for e in self.edges:
            self.in_edges[e["dst"]].append(e)
            self.out_edges[e["src"]].append(e)

        # Kahn topological sort
        in_degree = {n: len(self.in_edges[n]) for n in self.nodes}
        queue     = deque(n for n, d in in_degree.items() if d == 0)
        order     = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for e in self.out_edges[node]:
                in_degree[e["dst"]] -= 1
                if in_degree[e["dst"]] == 0:
                    queue.append(e["dst"])
        self.topo_order = order

    def _assign_vars(self):
        out_degree = {n: len(self.out_edges[n]) for n in self.nodes}

        var_at = {self.input_id: "x"}

        for node_id in self.topo_order:
            if node_id == self.input_id:
                continue

            my_in = self.in_edges[node_id]

            if not my_in:
                var_at[node_id] = "x"
                continue

            if len(my_in) > 1:
                # merge node: goes back to 'x' unless it also fans out
                var_at[node_id] = "x" if out_degree[node_id] <= 1 else node_id
                continue

            src = my_in[0]["src"]
            if out_degree[src] > 1:
                # branch start: use node_id so shared source tensor isn't overwritten
                var_at[node_id] = node_id
            else:
                # linear continuation: inherit source variable
                var_at[node_id] = var_at[src]

        self.var_at = var_at

    # ── __init__ emission ─────────────────────────────────────

    def _emit_init(self):
        lines = []
        for node_id in self.topo_order:
            info       = self.nodes[node_id]
            layer_type = info["type"]
            if layer_type in NO_MODULE_NODES:
                continue
            pytorch_class, param_name_map = PYTORCH_MAP[layer_type]
            params_str = self._build_params(layer_type, info["params"], param_name_map)
            lines.append(f"        self.{node_id} = {pytorch_class}({params_str})")
        return lines

    def _build_params(self, layer_type, params, param_name_map):
        parts = []
        for dsl_name, pytorch_name in param_name_map.items():
            if dsl_name in params:
                val = params[dsl_name]["value"]
                if isinstance(val, str):
                    parts.append(f'{pytorch_name}="{val}"')
                else:
                    parts.append(f"{pytorch_name}={val}")
        if layer_type == "MultiHeadAttn":
            parts.append("batch_first=True")
        return ", ".join(parts)

    # ── forward() emission ────────────────────────────────────

    def _emit_forward(self):
        lines = []
        for node_id in self.topo_order:
            lines.extend(self._emit_node(node_id))
        lines.append(f"        return {self.var_at[self.output_id]}")
        return lines

    def _emit_node(self, node_id):
        info       = self.nodes[node_id]
        layer_type = info["type"]
        my_in      = self.in_edges[node_id]
        out_var    = self.var_at[node_id]

        if layer_type == "__input__":
            return []

        # ── special ops (no self.x module) ────────────────────

        if layer_type == "Residual":
            shortcut_edges = [e for e in my_in if e.get("label") == "shortcut"]
            main_edges     = [e for e in my_in if e.get("label") != "shortcut"]
            main_var  = self.var_at[main_edges[0]["src"]]
            skip_var  = self.var_at[shortcut_edges[0]["src"]]
            return [f"        {out_var} = {main_var} + {skip_var}"]

        if layer_type == "Add":
            in_vars = [self.var_at[e["src"]] for e in my_in]
            rhs = " + ".join(in_vars)
            return [f"        {out_var} = {rhs}"]

        if layer_type == "Concat":
            in_vars = [self.var_at[e["src"]] for e in my_in]
            dim     = info["params"].get("dim", {}).get("value", 1)
            return [f"        {out_var} = torch.cat([{', '.join(in_vars)}], dim={dim})"]

        if layer_type == "Split":
            in_var  = self.var_at[my_in[0]["src"]]
            chunks  = info["params"].get("chunks", {}).get("value", 2)
            dim     = info["params"].get("dim", {}).get("value", 1)
            return [f"        parts = torch.chunk({in_var}, {chunks}, dim={dim})"]

        # ── nodes with pytorch modules ─────────────────────────

        if len(my_in) > 1:
            # implicit residual: node has a module but takes 2+ in-edges
            # one in-edge is labeled "residual_*" (skip); the other is the main path
            residual_edges = [e for e in my_in if (e.get("label") or "").startswith("residual")]
            main_edges     = [e for e in my_in if e not in residual_edges]
            if residual_edges:
                skip_var = self.var_at[residual_edges[0]["src"]]
                main_var = self.var_at[main_edges[0]["src"]]
                return [f"        {out_var} = self.{node_id}({skip_var} + {main_var})"]
            # fallback: just use first in-edge
            in_var = self.var_at[my_in[0]["src"]]
            return [f"        {out_var} = self.{node_id}({in_var})"]

        in_var = self.var_at[my_in[0]["src"]]

        if layer_type == "MultiHeadAttn":
            return [f"        {out_var}, _ = self.{node_id}({in_var}, {in_var}, {in_var})"]

        if layer_type in ("LSTM", "GRU"):
            return [f"        {out_var}, _ = self.{node_id}({in_var})"]

        return [f"        {out_var} = self.{node_id}({in_var})"]

    # ── assembly ───────────────────────────────────────────────

    def _assemble(self, init_lines, forward_lines):
        device     = self.config.get("device", "cpu")
        batch_size = self.config.get("batch_size", 1)
        shape_str  = ", ".join(str(s) for s in (batch_size,) + self.input_shape)

        parts = [
            "import torch",
            "import torch.nn as nn",
            "",
            "",
            f"class {self.model_name}(nn.Module):",
            f"    def __init__(self):",
            f"        super({self.model_name}, self).__init__()",
        ]
        parts.extend(init_lines)
        parts += [
            "",
            "    def forward(self, x):",
        ]
        parts.extend(forward_lines)
        parts += [
            "",
            "",
            "if __name__ == '__main__':",
            f"    device = torch.device('{device}')",
            f"    model = {self.model_name}().to(device)",
            f"    x = torch.randn({shape_str}).to(device)",
            f"    print(model(x).shape)",
        ]
        return "\n".join(parts) + "\n"
