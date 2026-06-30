import sys
from collections import deque

from antlr4 import ParseTreeWalker

from gen.NNGraphListener import NNGraphListener
from gen.NNGraphParser import NNGraphParser
from required_code_collection.ast import AST
from required_code_collection.make_ast_subtree import make_ast_subtree


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

NO_MODULE_NODES = {"Add", "Concat", "Residual", "Split"}

QUANT_MODES = {"dynamic", "static", "qat"}
UNIVERSAL_PARAMS = {"quant": str}


def error(msg, ctx):
    line = ctx.start.line
    col  = ctx.start.column
    print(f"[line {line}:{col}] SemanticError: {msg}", file=sys.stderr)
    raise SystemExit(1)

class NNGraphCustomListener(NNGraphListener):

    def __init__(self):
        self.model_name  = None
        self.input_id    = None
        self.input_shape = None
        self.output_id   = None
        self.nodes       = {}   # id -> {"type": str, "params": dict, "line": int}
        self.edges       = []   # list of {"src": str, "dst": str, "label": str|None, "line": int}
        self.config      = {}

        self.ast        = AST()
        self.rule_names = []

    # ── value / shape propagation ──────────────────────────────

    def exitShape_expr(self, ctx: NNGraphParser.Shape_exprContext):
        ints = [int(ctx.INT_LITERAL(i).getText()) for i in range(ctx.INT_LITERAL().__len__())]
        ctx.pvalue = tuple(ints)
        ctx.ptype  = tuple

    def exitValue(self, ctx: NNGraphParser.ValueContext):
        if ctx.FLOAT_LITERAL():
            ctx.pvalue = float(ctx.FLOAT_LITERAL().getText())
            ctx.ptype  = float
        elif ctx.INT_LITERAL():
            ctx.pvalue = int(ctx.INT_LITERAL().getText())
            ctx.ptype  = int
        elif ctx.BOOL_LITERAL():
            ctx.pvalue = ctx.BOOL_LITERAL().getText() == 'true'
            ctx.ptype  = bool
        elif ctx.STRING():
            ctx.pvalue = ctx.STRING().getText().strip('"')
            ctx.ptype  = str
        elif ctx.NONE():
            ctx.pvalue = None
            ctx.ptype  = type(None)
        elif ctx.shape_expr():
            ctx.pvalue = ctx.shape_expr().pvalue
            ctx.ptype  = tuple

    # ── model block ────────────────────────────────────────────

    def exitModel_block(self, ctx: NNGraphParser.Model_blockContext):
        self.model_name = ctx.ID().getText()

    def exitInput_decl(self, ctx: NNGraphParser.Input_declContext):
        self.input_id    = ctx.ID().getText()
        self.input_shape = ctx.shape_expr().pvalue
        # register input as a pseudo-node so edge validation finds it
        self.nodes[self.input_id] = {"type": "__input__", "params": {}, "line": ctx.start.line}

    def exitOutput_decl(self, ctx: NNGraphParser.Output_declContext):
        self.output_id = ctx.ID().getText()

    # ── graph block ────────────────────────────────────────────

    def exitParam(self, ctx: NNGraphParser.ParamContext):
        ctx.param_name  = ctx.ID().getText()
        ctx.param_value = ctx.value().pvalue
        ctx.param_type  = ctx.value().ptype

    def exitLayer_expr(self, ctx: NNGraphParser.Layer_exprContext):
        ctx.layer_type = ctx.ID().getText()
        ctx.params     = {}
        if ctx.param_list():
            for p in ctx.param_list().param():
                ctx.params[p.param_name] = {"value": p.param_value, "type": p.param_type}

    def exitNode_decl(self, ctx: NNGraphParser.Node_declContext):
        node_id    = ctx.ID().getText()
        layer_expr = ctx.layer_expr()
        layer_type = layer_expr.layer_type
        params     = layer_expr.params

        if node_id in self.nodes:
            error(f"Duplicate node ID '{node_id}'.", ctx)

        if layer_type not in LAYER_SCHEMA:
            error(f"Unknown layer type '{layer_type}'.", ctx)

        schema = LAYER_SCHEMA[layer_type]
        combined = {**schema, **UNIVERSAL_PARAMS}
        for pname, pinfo in params.items():
            if pname not in combined:
                error(f"Unknown parameter '{pname}' for layer '{layer_type}'.", ctx)
            expected = combined[pname]
            actual   = pinfo["type"]
            if isinstance(expected, tuple):
                if actual not in expected:
                    error(
                        f"Parameter '{pname}' of '{layer_type}' expects {expected}, got {actual.__name__}.",
                        ctx,
                    )
            else:
                if actual is not expected:
                    error(
                        f"Parameter '{pname}' of '{layer_type}' expects {expected.__name__}, got {actual.__name__}.",
                        ctx,
                    )

        if "quant" in params and params["quant"]["value"] not in QUANT_MODES:
            error(
                f"Unknown quant mode '{params['quant']['value']}'. Valid: {sorted(QUANT_MODES)}.",
                ctx,
            )

        self.nodes[node_id] = {"type": layer_type, "params": params, "line": ctx.start.line}

    def exitEdge_decl(self, ctx: NNGraphParser.Edge_declContext):
        ids   = ctx.ID()
        src   = ids[0].getText()
        dst   = ids[1].getText()
        label = None
        if ctx.STRING():
            label = ctx.STRING().getText().strip('"')

        if src not in self.nodes:
            declared = list(self.nodes.keys())
            error(
                f"Edge references undefined node '{src}'.\n  Hint: Declared nodes are: {declared}",
                ctx,
            )
        if dst not in self.nodes:
            declared = list(self.nodes.keys())
            error(
                f"Edge references undefined node '{dst}'.\n  Hint: Declared nodes are: {declared}",
                ctx,
            )

        self.edges.append({"src": src, "dst": dst, "label": label, "line": ctx.start.line})

    def exitConfig_entry(self, ctx: NNGraphParser.Config_entryContext):
        key = ctx.ID().getText()
        val = ctx.value().pvalue
        self.config[key] = val

    def exitGraph_block(self, ctx: NNGraphParser.Graph_blockContext):
        self._validate_output_declared(ctx)
        self._validate_orphans(ctx)
        self._validate_residual_arity(ctx)
        self._validate_no_cycles(ctx)
        self._validate_reachability(ctx)

    # ── AST construction (same contract as EVM) ────────────────

    def exitEveryRule(self, ctx):
        if not self.rule_names:
            return
        rule_name = self.rule_names[ctx.getRuleIndex()]
        make_ast_subtree(self.ast, ctx, rule_name)

    def exitStart(self, ctx: NNGraphParser.StartContext):
        if self.rule_names:
            make_ast_subtree(self.ast, ctx, "start", keep_node=True)

    # ── validation passes ──────────────────────────────────────

    def _validate_output_declared(self, ctx):
        if self.output_id not in self.nodes:
            # output node must be declared in graph
            error(
                f"Output node '{self.output_id}' is not declared in the graph block.",
                ctx,
            )

    def _validate_orphans(self, ctx):
        referenced = set()
        for e in self.edges:
            referenced.add(e["src"])
            referenced.add(e["dst"])
        for node_id, info in self.nodes.items():
            if info["type"] == "__input__":
                continue
            if node_id not in referenced:
                class _FakeCtx:
                    class start:
                        line   = info["line"]
                        column = 0
                error(f"Node '{node_id}' is declared but never referenced in any edge.", _FakeCtx)

    def _validate_residual_arity(self, ctx):
        in_edges = {}
        for node_id in self.nodes:
            in_edges[node_id] = []
        for e in self.edges:
            in_edges[e["dst"]].append(e)
        for node_id, info in self.nodes.items():
            if info["type"] == "Residual":
                count = len(in_edges[node_id])
                if count != 2:
                    class _FakeCtx:
                        class start:
                            line   = info["line"]
                            column = 0
                    error(
                        f"Node '{node_id}' (Residual) has {count} incoming edge(s); expected exactly 2.",
                        _FakeCtx,
                    )

    def _validate_no_cycles(self, ctx):
        in_degree = dict.fromkeys(self.nodes, 0)
        adj       = {n: [] for n in self.nodes}
        for e in self.edges:
            adj[e["src"]].append(e["dst"])
            in_degree[e["dst"]] += 1

        queue    = deque(n for n, d in in_degree.items() if d == 0)
        visited  = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for nxt in adj[node]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if visited != len(self.nodes):
            remaining = [n for n, d in in_degree.items() if d > 0]
            error(
                f"Cycle detected involving nodes: {remaining}.\n  Hint: NNGraph graphs must be acyclic DAGs.",
                ctx,
            )

    def _validate_reachability(self, ctx):
        adj = {n: [] for n in self.nodes}
        for e in self.edges:
            adj[e["src"]].append(e["dst"])

        visited = set()
        queue   = deque([self.input_id])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for nxt in adj[node]:
                queue.append(nxt)

        for node_id, info in self.nodes.items():
            if node_id not in visited and info["type"] != "__input__":
                class _FakeCtx:
                    class start:
                        line   = info["line"]
                        column = 0
                error(f"Node '{node_id}' is not reachable from input '{self.input_id}'.", _FakeCtx)

        if self.output_id not in visited:
            error(
                f"Output node '{self.output_id}' is not reachable from input '{self.input_id}'.",
                ctx,
            )
