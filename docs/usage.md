# Usage

## Install

Python 3.9 or later. ANTLR4 runtime is the only non-standard dependency:

```bash
pip install antlr4-python3-runtime==4.13.*
```

PyTorch is needed only to run the generated code, not to compile `.nng` files:

```bash
pip install torch
```

Java 11 or later is needed only if you modify `NNGraph.g4` and need to regenerate the
lexer and parser. It is not needed at runtime.

---

## CLI

Entry point: `main.py`. All flags:

```
python main.py [options]

  -i / --input PATH      .nng source file (default: input/mlp.nng)
  -o / --output PATH     output .py file  (default: output/model.py)
  --check-only           validate only; no code is emitted
  --print                print generated code to stdout
  --show-ast             visualize the AST using networkx
  --show-traversal       print topological traversal order to console
  --show-shapes          print inferred tensor shapes per node
  --graph-viz DOT_FILE   output Graphviz DOT file
  --onnx ONNX_FILE       export model to ONNX format
```

### Compile to a file

```bash
python main.py -i input/mlp.nng -o output/mlp.py
```

Prints `Written: output/mlp.py` to stderr on success. Creates the output directory
if it does not exist.

### Validate without generating

```bash
python main.py -i input/resblock.nng --check-only
```

Prints `OK  input/resblock.nng` to stderr. Returns exit code 0. No output file is
written even if `-o` is also passed.

### Print to stdout

```bash
python main.py -i input/inception.nng --print
```

Sends the generated Python source to stdout. Useful for piping into another tool or
for inspecting generated code without writing a file.

### Show inferred shapes

```bash
python main.py -i input/mlp.nng --show-shapes
```

Propagates tensor shapes through the graph and prints each node's output shape to
stderr. Catches dimension mismatches at compile time.

### Export Graphviz visualization

```bash
python main.py -i input/inception.nng --graph-viz output/inception.dot
```

Writes a DOT file with color-coded nodes (by layer type), shape annotations, and
dashed labeled edges. Render with `dot -Tpng output/inception.dot -o output/inception.png`.

### Export to ONNX

```bash
python main.py -i input/mlp.nng --onnx output/mlp.onnx
```

Generates the model code, instantiates it, runs `torch.onnx.export` with dynamic
batch axis, and saves the `.onnx` file.

### Combine flags

```bash
python main.py -i input/transformer.nng -o output/transformer.py --print
```

Writes the file and also prints to stdout in the same run.

---

## Run the generated model

The generated file is self-contained. It defines the model class and a `__main__`
block that runs a forward pass and prints the output shape:

```bash
python output/mlp.py
# → torch.Size([64, 10])
```

The `batch_size` and `device` values come from the `config` block of the `.nng`
source. If the `config` block is absent, the generated code defaults to `batch_size=1`
and `device='cpu'`.

---

## Python API

`compile_nng` is importable:

```python
from main import compile_nng

# compile and write file
compile_nng("input/mlp.nng", output_path="output/mlp.py")

# validate only
compile_nng("input/mlp.nng", check_only=True)

# return generated code as a string
code = compile_nng("input/mlp.nng")
```

Signature:

```python
def compile_nng(
    input_path: str,
    output_path: str | None = None,
    check_only: bool = False,
    print_output: bool = False,
    show_ast_flag: bool = False,
    show_traversal: bool = False,
    show_shapes: bool = False,
    graph_viz: str | None = None,
    onnx_path: str | None = None,
) -> str | None:
```

Returns the generated Python source string, or `None` when `check_only=True`.
Raises `SystemExit(1)` on any parse or semantic error.

---

## Quick-start

Compile the MLP example and run it:

```bash
cd /home/emmwhy/Projects/NNGraphDSL
python main.py -i input/mlp.nng -o output/mlp.py
python output/mlp.py
```

Expected output:

```
Written: output/mlp.py
torch.Size([64, 10])
```


