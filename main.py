import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from antlr4 import FileStream, CommonTokenStream, ParseTreeWalker
import argparse

from gen.NNGraphLexer import NNGraphLexer
from gen.NNGraphParser import NNGraphParser
from custom_listener import NNGraphCustomListener
from code_generator import CodeGenerator
from required_code_collection.ast_to_networkx_graph import show_ast
from required_code_collection.traverse_graph import topological_sort
from shape_inference import infer_shapes
from graph_viz import generate_dot
from onnx_export import export_onnx


def compile_nng(
    input_path,
    output_path=None,
    check_only=False,
    print_output=False,
    show_ast_flag=False,
    show_traversal=False,
    show_shapes=False,
    graph_viz=None,
    onnx_path=None,
):
    """
    Parse, semantically analyse, and optionally emit PyTorch code for a .nng file.
    Returns the generated source string (or None when check_only=True).
    Raises SystemExit on any error.
    """
    stream = FileStream(input_path, encoding="utf8")
    lexer = NNGraphLexer(stream)
    token_stream = CommonTokenStream(lexer)

    parser = NNGraphParser(token_stream)
    parse_tree = parser.start()

    if parser.getNumberOfSyntaxErrors() > 0:
        print(
            f"[ERROR] {parser.getNumberOfSyntaxErrors()} syntax error(s) in '{input_path}'. Aborting.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    listener = NNGraphCustomListener()
    listener.rule_names = parser.ruleNames
    walker = ParseTreeWalker()
    walker.walk(listener, parse_tree)

    if show_ast_flag:
        show_ast(listener.ast.root)

    traversal = topological_sort(listener.nodes, listener.edges)

    if show_traversal:
        print("Code gen traversal:", " → ".join(traversal), file=sys.stderr)

    shapes = infer_shapes(
        listener.nodes,
        listener.edges,
        listener.input_id,
        listener.input_shape,
        traversal,
    )

    if show_shapes:
        print("── Shape inference ──", file=sys.stderr)
        for nid in traversal:
            if nid != listener.input_id:
                print(f"  {nid}: {shapes[nid]}", file=sys.stderr)

    if graph_viz:
        dot = generate_dot(
            listener.nodes,
            listener.edges,
            listener.input_id,
            listener.output_id,
            listener.model_name,
            shapes,
        )
        with open(graph_viz, "w") as f:
            f.write(dot)
        print(f"Graph written: {graph_viz}", file=sys.stderr)

    code_gen = CodeGenerator()
    code_gen.load_from_listener(listener, traversal)

    if check_only:
        print(f"OK  {input_path}", file=sys.stderr)
        return None

    final_code = code_gen.generate()

    if print_output:
        print(final_code)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(final_code)
        print(f"Written: {output_path}", file=sys.stderr)

    if onnx_path:
        device = listener.config.get("device", "cpu")
        batch_size = listener.config.get("batch_size", 1)
        export_onnx(
            final_code,
            listener.model_name,
            listener.input_shape,
            onnx_path,
            device=device,
            batch_size=batch_size,
        )

    return final_code


def main(arguments):
    compile_nng(
        input_path=arguments.input,
        output_path=arguments.output if not arguments.check_only else None,
        check_only=arguments.check_only,
        print_output=arguments.print_output,
        show_ast_flag=arguments.show_ast,
        show_traversal=arguments.show_traversal,
        show_shapes=arguments.show_shapes,
        graph_viz=arguments.graph_viz,
        onnx_path=arguments.onnx,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NNGraph DSL → PyTorch compiler")
    ap.add_argument("-i", "--input", default="input/mlp.nng", help="Input .nng file")
    ap.add_argument("-o", "--output", default="output/model.py", help="Output .py file")
    ap.add_argument(
        "--check-only", action="store_true", help="Validate only; no code emitted"
    )
    ap.add_argument(
        "--print",
        dest="print_output",
        action="store_true",
        help="Print generated code to stdout",
    )
    ap.add_argument(
        "--show-ast", action="store_true", help="Visualize the AST using networkx"
    )
    ap.add_argument(
        "--show-traversal",
        action="store_true",
        help="Print topological traversal order to console",
    )
    ap.add_argument(
        "--show-shapes",
        action="store_true",
        help="Print inferred tensor shapes per node",
    )
    ap.add_argument("--graph-viz", metavar="DOT_FILE", help="Output Graphviz DOT file")
    ap.add_argument("--onnx", metavar="ONNX_FILE", help="Export model to ONNX format")
    main(ap.parse_args())
