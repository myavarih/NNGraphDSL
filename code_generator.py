from collections import deque
NO_MODULE_NODES = {"Add", "Concat", "Residual", "Split", "__input__"}
UNIVERSAL_PARAMS = {"quant"}
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
        self._graph_computed = False
    def load_from_listener(self, listener, traversal=None):
        self.model_name  = listener.model_name
        self.input_id    = listener.input_id
        self.input_shape = listener.input_shape
        self.output_id   = listener.output_id
        self.nodes       = listener.nodes
        self.edges       = listener.edges
        self.config      = listener.config
        if traversal is not None:
            self.topo_order = traversal
            self._graph_computed = True
            self.in_edges  = {n: [] for n in self.nodes}
            self.out_edges = {n: [] for n in self.nodes}
            for e in self.edges:
                self.in_edges[e["dst"]].append(e)
                self.out_edges[e["src"]].append(e)
    def generate(self):
        self._compute_graph()
        self._assign_vars()
        init_lines    = self._emit_init()
        forward_lines = self._emit_forward()
        return self._assemble(init_lines, forward_lines)
    def _compute_graph(self):
        if self._graph_computed:
            return
        self.in_edges  = {n: [] for n in self.nodes}
        self.out_edges = {n: [] for n in self.nodes}
        for e in self.edges:
            self.in_edges[e["dst"]].append(e)
            self.out_edges[e["src"]].append(e)
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
        self._graph_computed = True
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
                var_at[node_id] = "x" if out_degree[node_id] <= 1 else node_id
                continue
            src = my_in[0]["src"]
            src_type = self.nodes[src]["type"]
            if src_type == "Split":
                idx = self.out_edges[src].index(my_in[0])
                var_at[node_id] = f"{src}_{idx}"
            elif out_degree[src] > 1:
                var_at[node_id] = node_id
            else:
                var_at[node_id] = var_at[src]
        self.var_at = var_at
    def _input_var(self, edge):
        """Resolve the variable name for an edge's source, handling Split chunk indexing."""
        src = edge["src"]
        return self.var_at[src]
    def _emit_init(self):
        lines = []
        has_quant = False
        for node_id in self.topo_order:
            info       = self.nodes[node_id]
            layer_type = info["type"]
            if layer_type in NO_MODULE_NODES:
                continue
            pytorch_class, param_name_map = PYTORCH_MAP[layer_type]
            params_str = self._build_params(layer_type, info["params"], param_name_map)
            quant_mode = info["params"].get("quant", {}).get("value")
            if quant_mode == "qat":
                lines.append(f"        self.{node_id} = torch.quantization.QuantWrapper({pytorch_class}({params_str}))")
                has_quant = True
            else:
                lines.append(f"        self.{node_id} = {pytorch_class}({params_str})")
                if quant_mode:
                    has_quant = True
        if has_quant:
            lines.append("        self.quant = torch.quantization.QuantStub()")
            lines.append("        self.dequant = torch.quantization.DeQuantStub()")
        self._has_quant = has_quant
        return lines
    def _build_params(self, layer_type, params, param_name_map):
        parts = []
        for dsl_name, pytorch_name in param_name_map.items():
            if dsl_name in params and dsl_name not in UNIVERSAL_PARAMS:
                val = params[dsl_name]["value"]
                if isinstance(val, str):
                    parts.append(f'{pytorch_name}="{val}"')
                else:
                    parts.append(f"{pytorch_name}={val}")
        if layer_type == "MultiHeadAttn":
            parts.append("batch_first=True")
        return ", ".join(parts)
    def _emit_forward(self):
        lines = []
        if getattr(self, '_has_quant', False):
            lines.append("        x = self.quant(x)")
        for node_id in self.topo_order:
            lines.extend(self._emit_node(node_id))
        if getattr(self, '_has_quant', False):
            lines.append(f"        {self.var_at[self.output_id]} = self.dequant({self.var_at[self.output_id]})")
        lines.append(f"        return {self.var_at[self.output_id]}")
        return lines
    def _emit_node(self, node_id):
        info       = self.nodes[node_id]
        layer_type = info["type"]
        my_in      = self.in_edges[node_id]
        out_var    = self.var_at[node_id]
        if layer_type == "__input__":
            return []
        if layer_type == "Residual":
            shortcut_edges = [e for e in my_in if e.get("label") == "shortcut"]
            main_edges     = [e for e in my_in if e.get("label") != "shortcut"]
            main_var  = self._input_var(main_edges[0])
            skip_var  = self._input_var(shortcut_edges[0])
            return [f"        {out_var} = {main_var} + {skip_var}"]
        if layer_type == "Add":
            in_vars = [self._input_var(e) for e in my_in]
            rhs = " + ".join(in_vars)
            return [f"        {out_var} = {rhs}"]
        if layer_type == "Concat":
            in_vars = [self._input_var(e) for e in my_in]
            dim     = info["params"].get("dim", {}).get("value", 1)
            return [f"        {out_var} = torch.cat([{', '.join(in_vars)}], dim={dim})"]
        if layer_type == "Split":
            in_var  = self._input_var(my_in[0])
            chunks  = info["params"].get("chunks", {}).get("value", 2)
            dim     = info["params"].get("dim", {}).get("value", 1)
            chunk_vars = ", ".join(f"{node_id}_{i}" for i in range(chunks))
            return [f"        {chunk_vars} = torch.chunk({in_var}, {chunks}, dim={dim})"]
        if len(my_in) > 1:
            residual_edges = [e for e in my_in if (e.get("label") or "").startswith("residual")]
            main_edges     = [e for e in my_in if e not in residual_edges]
            if residual_edges:
                skip_var = self._input_var(residual_edges[0])
                main_var = self._input_var(main_edges[0])
                return [f"        {out_var} = self.{node_id}({skip_var} + {main_var})"]
            in_var = self._input_var(my_in[0])
            return [f"        {out_var} = self.{node_id}({in_var})"]
        in_var = self._input_var(my_in[0])
        if layer_type == "MultiHeadAttn":
            return [f"        {out_var}, _ = self.{node_id}({in_var}, {in_var}, {in_var})"]
        if layer_type in ("LSTM", "GRU"):
            return [f"        {out_var}, _ = self.{node_id}({in_var})"]
        return [f"        {out_var} = self.{node_id}({in_var})"]
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
        parts.extend(self._emit_training(shape_str))
        return "\n".join(parts) + "\n"
    def _emit_training(self, shape_str):
        loss_fn = self.config.get("loss")
        if not loss_fn:
            return []
        optimizer = self.config.get("optimizer", "Adam")
        lr        = self.config.get("lr", 0.001)
        epochs    = self.config.get("epochs", 10)
        LOSS_MAP = {
            "CrossEntropyLoss": "nn.CrossEntropyLoss()",
            "MSELoss":          "nn.MSELoss()",
            "BCELoss":          "nn.BCELoss()",
            "BCEWithLogitsLoss": "nn.BCEWithLogitsLoss()",
            "L1Loss":           "nn.L1Loss()",
            "NLLLoss":          "nn.NLLLoss()",
            "SmoothL1Loss":     "nn.SmoothL1Loss()",
        }
        OPTIM_MAP = {
            "Adam":     "torch.optim.Adam",
            "SGD":      "torch.optim.SGD",
            "AdamW":    "torch.optim.AdamW",
            "RMSprop":  "torch.optim.RMSprop",
        }
        loss_expr = LOSS_MAP.get(loss_fn, f"nn.{loss_fn}()")
        optim_cls = OPTIM_MAP.get(optimizer, f"torch.optim.{optimizer}")
        return [
            "",
            f"    criterion = {loss_expr}",
            f"    optimizer = {optim_cls}(model.parameters(), lr={lr})",
            "",
            f"    # Training loop",
            f"    model.train()",
            f"    for epoch in range({epochs}):",
            f"        x = torch.randn({shape_str}).to(device)",
            f"        y = model(x)",
            f"        target = torch.zeros_like(y)",
            f"        loss = criterion(y, target)",
            f"        optimizer.zero_grad()",
            f"        loss.backward()",
            f"        optimizer.step()",
            f"        print(f'Epoch { epoch+1} /{epochs}, Loss: { loss.item():.4f} ')",
        ]
