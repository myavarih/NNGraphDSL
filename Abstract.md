Abstract: 

NNGraph DSL is a domain-specific language designed to describe neural network architectures as directed graphs. It accepts .nng files as input and produces executable PyTorch code consisting of a complete nn.Module subclass with \__init__ and forward methods (plus possibly a training loop). The compiler is built with ANTLR4 version 4.13.2

Tensor shape inference, Graphviz graph visualization, training scaffolding, quantization annotations, and ONNX export are also supported.





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







Grammar:

the grammar is simple and short, Complete grammar rules:







Type System:



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

   at the end back in _emit_forward we check and if any node had quant param in the very begining we add a ` x = self.quant(x)` line and the end use dequant like this:
```python
{self.var_at[self.output_id]} = self.dequant({self.var_at[self.output_id]})
```

5. _assemble: this first generates the import and class declaration boilerplate then adds init lines and forward lines (from the last 2 steps) and in the end calls _emit_training to optionally generate training loop

6. _emit_training: we first check and if loss function is not specified we will return empty (no training loop). if not we simply map the grammar values (like optimizer, lr and epochs) to PyTorch syntax and place them in the straight forward training loop boilerplate

examples:
**MLP Input (`input/mlp.nng`)**:
```nngraph
model MLP {
    input x : tensor(784)
    output out
}

graph {
    node fc1   : Linear(in_features=784, out_features=256)
    node relu1 : ReLU()
    node drop1 : Dropout(p=0.3)
    node fc2   : Linear(in_features=256, out_features=128)
    node relu2 : ReLU()
    node fc3   : Linear(in_features=128, out_features=10)
    node out   : Softmax(dim=1)

    edge x -> fc1
    edge fc1 -> relu1
    edge relu1 -> drop1
    edge drop1 -> fc2
    edge fc2 -> relu2
    edge relu2 -> fc3
    edge fc3 -> out
}

config {
    batch_size = 64
    device = "cuda"
    loss = "MSELoss"
    optimizer = "Adam"
    lr = 0.001
    epochs = 5
}
```

**MLP Output (`output/mlp.py`)**:
```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=256)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=128, out_features=10)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda')
    model = MLP().to(device)
    x = torch.randn(64, 784).to(device)
    print(model(x).shape)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(5):
        x = torch.randn(64, 784).to(device)
        y = model(x)
        target = torch.zeros_like(y)
        loss = criterion(y, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/5, Loss: {loss.item():.4f}')
```

**ResBlock Input (`input/resblock.nng`)**:
```nngraph
model ResBlock {
    input x : tensor(64, 32, 32)
    output out
}

graph {
    node conv1 : Conv2d(in_ch=64, out_ch=64, kernel=3, stride=1, padding=1)
    node bn1   : BatchNorm2d(num_features=64)
    node relu1 : ReLU()
    node conv2 : Conv2d(in_ch=64, out_ch=64, kernel=3, stride=1, padding=1)
    node bn2   : BatchNorm2d(num_features=64)
    node skip  : Residual()
    node relu2 : ReLU()
    node out   : Flatten()

    edge x -> conv1
    edge conv1 -> bn1
    edge bn1 -> relu1
    edge relu1 -> conv2
    edge conv2 -> bn2
    edge bn2 -> skip
    edge x -> skip [label="shortcut"]
    edge skip -> relu2
    edge relu2 -> out
}

config {
    batch_size = 32
    device = "cpu"
    loss = "MSELoss"
    optimizer = "AdamW"
    lr = 0.0005
    epochs = 3
}
```

**ResBlock Output (`output/resblock.py`)**:
```python
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.relu2 = nn.ReLU()
        self.out = nn.Flatten()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.relu1(conv1)
        conv1 = self.conv2(conv1)
        conv1 = self.bn2(conv1)
        x = conv1 + x
        x = self.relu2(x)
        x = self.out(x)
        return x

if __name__ == '__main__':
    device = torch.device('cpu')
    model = ResBlock().to(device)
    x = torch.randn(32, 64, 32, 32).to(device)
    print(model(x).shape)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)

    # Training loop
    model.train()
    for epoch in range(3):
        x = torch.randn(32, 64, 32, 32).to(device)
        y = model(x)
        target = torch.zeros_like(y)
        loss = criterion(y, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/3, Loss: {loss.item():.4f}')
```

**QuantMLP Input (`input/quant_mlp.nng`)**:
```nngraph
model QuantMLP {
    input x : tensor(784)
    output out
}

graph {
    node fc1   : Linear(in_features=784, out_features=128, quant="qat")
    node relu1 : ReLU()
    node fc2   : Linear(in_features=128, out_features=10, quant="qat")
    node out   : Softmax(dim=1)

    edge x -> fc1
    edge fc1 -> relu1
    edge relu1 -> fc2
    edge fc2 -> out
}

config {
    batch_size = 32
    device = "cpu"
    loss = "NLLLoss"
    optimizer = "Adam"
    lr = 0.001
    epochs = 2
}
```

**QuantMLP Output (`output/quant_mlp.py`)**:
```python
import torch
import torch.nn as nn

class QuantMLP(nn.Module):
    def __init__(self):
        super(QuantMLP, self).__init__()
        self.fc1 = torch.quantization.QuantWrapper(nn.Linear(in_features=784, out_features=128))
        self.relu1 = nn.ReLU()
        self.fc2 = torch.quantization.QuantWrapper(nn.Linear(in_features=128, out_features=10))
        self.out = nn.Softmax(dim=1)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.out(x)
        x = self.dequant(x)
        return x

if __name__ == '__main__':
    device = torch.device('cpu')
    model = QuantMLP().to(device)
    x = torch.randn(32, 784).to(device)
    print(model(x).shape)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(2):
        x = torch.randn(32, 784).to(device)
        y = model(x)
        target = torch.zeros_like(y)
        loss = criterion(y, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/2, Loss: {loss.item():.4f}')
```

**TransformerEncoder Input (`input/transformer.nng`)**:
```nngraph
model TransformerEncoder {
    input x : tensor(512, 128)
    output out
}

graph {
    node attn  : MultiHeadAttn(embed_dim=128, num_heads=8)
    node drop1 : Dropout(p=0.1)
    node norm1 : LayerNorm(normalized_shape=128)

    node ff1   : Linear(in_features=128, out_features=512)
    node gelu1 : GELU()
    node drop2 : Dropout(p=0.1)
    node ff2   : Linear(in_features=512, out_features=128)
    node norm2 : LayerNorm(normalized_shape=128)
    node out   : Flatten(start_dim=0, end_dim=1)

    edge x -> attn
    edge attn -> drop1
    edge drop1 -> norm1
    edge x -> norm1 [label="residual_1"]

    edge norm1 -> ff1
    edge ff1 -> gelu1
    edge gelu1 -> drop2
    edge drop2 -> ff2
    edge ff2 -> norm2
    edge norm1 -> norm2 [label="residual_2"]
    edge norm2 -> out
}
```

**TransformerEncoder Output (`output/transformer.py`)**:
```python
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
        self.drop1 = nn.Dropout(p=0.1)
        self.norm1 = nn.LayerNorm(normalized_shape=128)
        self.ff1 = nn.Linear(in_features=128, out_features=512)
        self.gelu1 = nn.GELU()
        self.drop2 = nn.Dropout(p=0.1)
        self.ff2 = nn.Linear(in_features=512, out_features=128)
        self.norm2 = nn.LayerNorm(normalized_shape=128)
        self.out = nn.Flatten(start_dim=0, end_dim=1)

    def forward(self, x):
        attn, _ = self.attn(x, x, x)
        attn = self.drop1(attn)
        norm1 = self.norm1(x + attn)
        ff1 = self.ff1(norm1)
        ff1 = self.gelu1(ff1)
        ff1 = self.drop2(ff1)
        ff1 = self.ff2(ff1)
        x = self.norm2(norm1 + ff1)
        x = self.out(x)
        return x

if __name__ == '__main__':
    device = torch.device('cpu')
    model = TransformerEncoder().to(device)
    x = torch.randn(1, 512, 128).to(device)
    print(model(x).shape)
```

Shape Inference:
Definition 4.5.1 ▶ Shape Inference
The shape_inference.py module propagates tensor shapes from the input node along the topological order. For each layer type, specific rules are applied: Linear changes the last dimension, Conv2d computes spatial dimensions, Flatten merges dimensions, and so on. 

When a dimension mismatch is detected, a ShapeError is reported with the line number of the node declaration, and compilation is halted. The --show-shapes flag displays each node’s output shape.

![image-20260702220712481](/home/emmwhy/.config/Typora/typora-user-images/image-20260702220712481.png) 

![image-20260702220817788](/home/emmwhy/.config/Typora/typora-user-images/image-20260702220817788.png)
Graphviz Visualization:

this inputs nodes, edges, input_id, output_id, model_name and shapes and generates a dot file with colors and stuff to make it beautiful!
not much logic here just making a pretty dot file

the `dot` tool can then be used to generate an image from the dot file

![inception](/home/emmwhy/Projects/NNGraphDSL/output/inception.png)

![transformer](/home/emmwhy/Projects/NNGraphDSL/output/transformer.png)





ONNX Export
The onnx_export.py module instantiates the generated code and uses torch.onnx.export
to produce an .onnx file with a dynamic batch axis and opset 17.



## Conclusion and Future Work

### Conclusion
The implementation of the NNGraph DSL demonstrates the feasibility of representing complex neural network topologies through a domain-specific language and automatically generating their PyTorch implementations.

### Future Work
- **Dynamic Control Flow:** Extending the DSL to support recurrent structures with dynamic lengths or conditional branching (e.g., if-else constructs).
- **Macros and Sub-graphs:** Allowing users to define reusable block macros (like an `InceptionBlock` macro) that can be instantiated multiple times.
