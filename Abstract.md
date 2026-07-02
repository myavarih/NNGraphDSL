Abstract: 

NNGraph DSL is a domain-specific language designed to describe neural network architectures as directed graphs. It accepts .nng files as input and produces executable PyTorch code consisting of a complete nn.Module subclass with \__init__ and forward methods (plus possibly a training loop). The compiler is built with ANTLR4 version 4.13.2

Tensor shape inference, Graphviz graph visualization, training scaffolding, quantization annotations, and ONNX export are also supported.

This Implementation also has a test suite made of 55 tests to ensure correct behavior.



Motivation:

Writing nn.Module PyTorch Codes by hand (or directly via LLM response) is repetitive, error prone and most importantly somehow unnecessary! so this project aims to automate and simplify this process by declaring a simple DSL called NNGraph so users can write this simpler syntax (or ask their AI tools to output in this syntax) instead of the more complex PyTorch syntax. this can be helpful for several reasons: 1. writing this syntax is easier for both Humans and AI Agents. 2. this syntax is less error prone as the compiler provides useful error checkings as well. 3. AI Agents can use less tokens if paired with this tool like this: the agent will output code in nng format then uses this tool to convert it into PyTorch Code to present to user, this makes the Agent less error prone (can catch it's error earlier) and also makes it use less tokens! 4. because of the error checkings and utilities this compiler provides programmers can also diagnose what's wrong easier and earlier. 5. this approach is memory optimized as it allocates minimum amount of variables needed that can be annoying and often forgotten to be handled by humans (and LLM agents). 6. this compiler provides easier generation of boilerplate codes, a good example is Quantization that we will examine later



Goals:
•Express any feed-forward or branching neural network topology as a graph.
•Generate clean, idiomatic PyTorch code 
•Support common layer types, activations, normalization, and skip connections.
•Be implementation-agnostic — the DSL is the single source of truth.
•Allow inline hyperparameter configuration per node.

note that the generated code is meant to be an starting point and it's clear that code can be further expanded as needed.



Scope:

The Implemention supports Code generation (nn.Module \__init__  , forward() training loop and setups) plus basic Semantic Checks and also Shape inference checks, debug and vizulizations tools, ONNX export and Quantization annotations.

but it does not support Dynamic control flow meaning the graph is static and known at model definition time.

we also do not have checks for loss function to match the architecture.

macros and dynamic control flow is planned as future work



Related work: 

The NADER Framework

NADER (Neural Architecture Design via Representation) is a framework that models
neural network architectures as directed computational graphs (DAGs). Each node
represents a layer operation and each edge represents tensor flow.
NNGraph DSL adopts this graph model as the foundation of its language design.



Tools and Tech Stack:
ANTLR4:

ANTLR4 (ANother Tool for Language Recognition) is a tool that automatically gen-
erates a lexer, parser, and listener/visitor from a .g4 grammar file. Version used:
ANTLR 4.13.2.

this was the main tool used to Implement the compiler

Python and Pytest:

pytest version 8.4.2

Python 3.14.3

PyTorch:

we didn't directly use PyTorch in this Project but as we are generating code the version 2.10.0 was used for Code Generation and Correctness verification





Architecture Overview:

The toolchain follows a classic compiler pipeline:

```
NNGraph Source (.nng)
        │
        ▼
┌─────────────┐
│   ANTLR4    │  Lexer + Parser (generated from NNGraph.g4)
└──────┬──────┘
       │ Parse Tree
       ▼
┌─────────────┐
│  Semantic   │  Type-checks nodes, validates edges, resolves shapes
│  Analyzer   │
└──────┬──────┘
       │ Validated AST
       ▼
┌─────────────┐
│    Code     │  Emits PyTorch Python module
│  Generator  │
└──────┬──────┘
       │
       ▼
Output PyTorch .py File
```



Language Design:

Program Strcuture:

Every .nng file consists of three blocks. The config block is optional:

<the code here stays the same>

<the 3 definitions stay the same>



Grammar:

the grammar is simple and short, Complete grammar rules:

<the code block stays the same>

<the important design note is removed>



Type System:

-table stays the same- but description col will be added from this table:

| DSL Type | Example | Description |
|---|---|---|
| `int` | `128` | Integer literal |
| `float` | `0.1` | Floating-point literal |
| `bool` | `true` / `false` | Boolean flag |
| `string` | `"relu"` | Quoted string |
| `shape` | `(3, 224, 224)` | Tuple of ints for tensor shape |
| `None` | `None` | Optional absent value |

note: The distinction between int and float is enforced at compile time. For example, Dropout(p=1)
produces an error because p expects a float, not an int.

layer types:

in parametrized layers the table col pythorch class needs to be pytorch mapping

make the title of graph operations Special operations and remove (no Module)

Edge Syntax: keep the existing explanation but also mention that labels are used for these purposes only in this implementation to improve clarity

-- then include comments after Edge Syntax

**Comments**

```
// This is a single-line comment

/* This is a
   multi-line comment */
```

then remove example from here



Implementation:

Compiler Architecture
Definition 4.1.1 ▶ Compilation pipeline
The compiler has six stages:

1. Parsing: .nng File → NNGraphLexer → NNGraphParser.program()
  → parse tree
2. Semantic analysis: ParseTreeWalker + NNGraphCustomListener →
  graph model (+ AST - which is not used apart from debuging)
3. Topological sort: compute execution order using Kahn’s algorithm
4. Shape inference: propagate tensor shapes through the graph and detect di-
  mension mismatches (optional: --show-shapes)
5. Code generation: CodeGenerator.generate() → Python string
6. Output: write to .py file (+ optional: Graphviz DOT output, or ONNX export)

then remove this part: File Responsibilities

add a section here: Parsing:

this part is done automatically by ANTLR4. 

here are some parse tree visualizations based on our grammar .g4 file from the ANTLR extension in PyCharm environment:

![image-20260702164342702](/home/emmwhy/.config/Typora/typora-user-images/image-20260702164342702.png)

![image-20260702164346991](/home/emmwhy/.config/Typora/typora-user-images/image-20260702164346991.png)

<img src="/home/emmwhy/Downloads/parseTree.png" alt="parseTree" style="zoom:100%;" />

Semantic Analyzer:

we use a customListener approach to analyze the code semantically. this works like this: we declare functions to be executed once we exit (wanna move on from traversing) each node using the tools provided for us from ANTLR4.

3 kinds of work is done:

1. populate ctx (context - ANTLR provides a ctx for each node with some info - we use those and add some more fields based on our needs) fields for each node (using it's children nodes or nodes that were traversed before in general) - and global lists and dictionaries - these will be used in kind 2 when doing the semantic checks (these will be used in Code Generation too and can be considered the output of Semantic Analysis Step)
2. do Semantic Checks and output errors if needed using the info from kind 1
3. AST Creation: AST will be created by exitEveryRule calling the make_ast_subtree on all nodes which creates the AST subtree, since AST is not crucial to this project we will not examine the details further

kind 1 is routine and therefore we skip explaining it in more detail.

the validations done in kind 2 are these:

1. check for duplicate node id: in exitNode_decl we check to confirm the node id is not already in nodes dictionary
2. check for unknown layer type: check to confirm the layer type is declared in our fixed LAYER_SCHEMA in exitNode_decl
3. check for correct parameter types: using LAYER_SCHEMA and UNIVERSAL_PARAMS (quant) also in exitNode_decl (this checks if each layer has correct parameters and the paramter values have correct type)
4. check for correct quant value: check to confirm the value of quant param is correct using QUANT_MODES also in exitNode_decl
5. check if the nodes exist in edge declaration: in exitEdge_decl
6. check for output declaration: after the graph block is completed -> we simple check if output is in nodes
7. check for orphan nodes: for each edge we make a refrenced flag fr both src and dst true and see if any node has a false flag after
8. check if residual nodes have exactly 2 inputs: we find residuals and check if they have 2 in_edges
9. cycle detection: we use Kahn's algorithm to deternine if the graph is a DAG: in each step we remove the node with no in_edges and then update the in_edges for other nodes untill the graph has no more nodes, if in any step there is no node withn0 in_edges the graph has a cycle
10. reachability: we use BFS to traverse the graph and  check if all nodes are reachable from input (including output)

note: items 6 to 10 are done in exitGraph_block after the graph block is walked and nodes and edges are filled

if in any case those validation fail we will have an error in this format:

Listing 4.2.3 ▶ Error format
1
2
[line L:C] SemanticError: <message>

example:
![image-20260702215847618](/home/emmwhy/.config/Typora/typora-user-images/image-20260702215847618.png)



<replace all of Semantic Analyzer section with what I said above>



Code Generator:

Code generation is done via the info gathered by the last step which is Semantic Analysis (like nodes and edges and model_name and ...)

it has 6 steps:

1. _compute_graph: uses the same Kahn's algorithm but this time for topological sort, we need topological sort because it's essential to first process the input nodes of a node then that node to able to generate the code correctly, the algorithm is simple too just like before but after removing a node we append it to the topo_order too

2. _assign_vars: this assigns varaibles to each node but uses 4 rules to ensure minimum memory allocation along with correct behaviour: 

   1. for input node always allocate x
   2. for merge nodes (in_degree > 1): if the out degree is 1 or 0 allocate x - if out degree is more than 1 allocate a variable named as 'node_id'
   3. Branch nodes (predecessor.out_degree > 1): allocate a variable named as node_id (split nodes' out nodes also get named similarly but with a small difference: the variable is named like this: source_name + '_' + index of the node)
   4. and if out_degree is 1 we will allocate the same variable allocated to it's predecessor to reuse memory

3. _emit_init: for this part we need to get pytorch class name for each node that has a module and also make it's params  string in correct format. we have a map to map each node (not in nodes with no modules like Add) to it's PyTorch class name and also get the names of params that it accepts. then with those param name we use _build_params to build the params string like this: if the value has type string wrap it in double qoutes if nor just simply set it like `param_name = param_value` and another exception is for MultiHeadAttn layer, we always add this param to it: `batch_first=True`.
   now we have the data we need, so for each node we first check if quantization is enabled, if yes we wrap the code into a PyTorch Quntization warper if not we emit the normal code, and in the end if any of the nodes had quantization enabled we emit code for Stubs too
   in this step topo order does not matter but we use topo order anyway

4. _emit_forward: this uses the helper _emit_node to emit the line for each node:
   first we examine the behavour for special nodes:
   Residual: 
   we determine the shortcut edge (which is the  edge that las the shortcut label) and also the main branch edge (which does not have shortcut label)
   then we get the ralated vars to the src of those edges and generate the code like: out_var (determined by getting var_at[current_node]) = main_var + shortcut_var


   Add:
   for Add we do a similar job: we add all input vars and assign back to the out_var

   Concat:
   this is similar to Add but we also dim from the params of that node and use torch.cat instaed of simple +

   Split:
   this is again similar to Concat but we need to get chunks and dim from params and use torch.chunk

   now for nodes that have Modules:
   one special case is implicit Residuals when the node has more than 1 input. we handle these like normal Residuals with the difference that we look for the edges that their label start with 'residual' to mark as shortcut
   and we give the addition of those main_var and shortcut_var to the module input
   if no such edges with residual label found then we fall-back to default behavior where we simply use the first edge's src's var as the input of the module

   normal case is when we have one input:
   but there is 2 special nodes in that case too:
   MultiHeadAttn which has a special line of code and also LSTM and GRU which have another special line
   if not in these types the normal line is emited

   at the end back in _emit_forward we check and if any node had quant param in the very begining we add a ` x = self.quant(x)` line and the end use dequant like this: `{self.var_at[self.output_id]} = self.dequant({self.var_at[self.output_id]})`

5. _assemble: this first generates the import and class declaration boilerplate then adds init lines and forward lines (from the last 2 steps) and in the end calls _emit_training to optionally generate training loop

6. _emit_training: we first check and if loss function is not specified we will return empty (no training loop). if not we simply map the grammar values (like optimizer, lr and epochs) to PyTorch syntax and place them in the straight forward training loop boilerplate

examples:
-- add examples from input and output dir --

Shape Inference:
Definition 4.5.1 ▶ Shape Inference
The shape_inference.py module propagates tensor shapes from the input node along the topological order. For each layer type, specific rules are applied: Linear changes the last dimension, Conv2d computes spatial dimensions, Flatten merges dimensions, and so on. 

When a dimension mismatch is detected, a ShapeError is reported with the line number of the node declaration, and compilation is halted. The --show-shapes flag displays each node’s output shape.

examples:

ShapeError:

![image-20260702220712481](/home/emmwhy/.config/Typora/typora-user-images/image-20260702220712481.png) 

Success:

![image-20260702220817788](/home/emmwhy/.config/Typora/typora-user-images/image-20260702220817788.png)

Graphviz Visualization:

this inputs nodes, edges, input_id, output_id, model_name and shapes and generates a dot file with colors and stuff to make it beautiful!
not much logic here just making a pretty dot file

the `dot` tool can then be used to generate an image from the dot file

examples:
inception: 

![inception](/home/emmwhy/Projects/NNGraphDSL/output/inception.png)

transformer:
![transformer](/home/emmwhy/Projects/NNGraphDSL/output/transformer.png)



-- then remove the trainingscaffold and quant sections---

ONNX Export
The onnx_export.py module instantiates the generated code and uses torch.onnx.export
to produce an .onnx file with a dynamic batch axis and opset 17.

-- remove the 2 next chapters Evaluation and design decisions--

Conclusion and Future Work:
-- remove all mention to tests and numbers like 14 parser rules and ... --
-- remove the implemented features section --

fix references the last link is cut of due to long length