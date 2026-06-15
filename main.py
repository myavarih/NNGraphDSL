import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from antlr4 import FileStream, CommonTokenStream, ParseTreeWalker
import argparse

from gen.NNGraphLexer import NNGraphLexer
from gen.NNGraphParser import NNGraphParser
from custom_listener import NNGraphCustomListener
from code_generator import CodeGenerator


def compile_nng(input_path, output_path=None, check_only=False, print_output=False):
    """
    Parse, semantically analyse, and optionally emit PyTorch code for a .nng file.
    Returns the generated source string (or None when check_only=True).
    Raises SystemExit on any error.
    """
    stream       = FileStream(input_path, encoding='utf8')
    lexer        = NNGraphLexer(stream)
    token_stream = CommonTokenStream(lexer)

    parser     = NNGraphParser(token_stream)
    parse_tree = parser.program()

    if parser.getNumberOfSyntaxErrors() > 0:
        print(f"[ERROR] {parser.getNumberOfSyntaxErrors()} syntax error(s) in '{input_path}'. Aborting.",
              file=sys.stderr)
        raise SystemExit(1)

    listener            = NNGraphCustomListener()
    listener.rule_names = parser.ruleNames
    walker              = ParseTreeWalker()
    walker.walk(listener, parse_tree)

    if check_only:
        print(f"OK  {input_path}", file=sys.stderr)
        return None

    code_gen = CodeGenerator()
    code_gen.load_from_listener(listener)
    final_code = code_gen.generate()

    if print_output:
        print(final_code)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(final_code)
        print(f"Written: {output_path}", file=sys.stderr)

    return final_code


def main(arguments):
    compile_nng(
        input_path   = arguments.input,
        output_path  = arguments.output if not arguments.check_only else None,
        check_only   = arguments.check_only,
        print_output = arguments.print_output,
    )


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='NNGraph DSL → PyTorch compiler')
    ap.add_argument('-i', '--input',        default='input/mlp.nng',   help='Input .nng file')
    ap.add_argument('-o', '--output',       default='output/model.py', help='Output .py file')
    ap.add_argument('--check-only',         action='store_true',       help='Validate only; no code emitted')
    ap.add_argument('--print',              dest='print_output',
                    action='store_true',    help='Print generated code to stdout')
    main(ap.parse_args())
