grammar NNGraph;

start: model_block graph_block config_block? EOF;

model_block: MODEL ID '{' input_decl output_decl '}';
input_decl:  INPUT ID ':' TENSOR shape_expr;
output_decl: OUTPUT ID;

graph_block: GRAPH '{' (node_decl | edge_decl)* '}';

node_decl:  NODE ID ':' layer_expr;
layer_expr: ID '(' param_list? ')';
param_list: param (',' param)*;
param:      ID '=' value;

edge_decl: EDGE ID ARROW ID ('[' LABEL '=' STRING ']')?;

config_block:  CONFIG '{' config_entry* '}';
config_entry:  ID '=' value;

value:      FLOAT_LITERAL
          | INT_LITERAL
          | BOOL_LITERAL
          | STRING
          | NONE
          | shape_expr;

shape_expr: '(' INT_LITERAL (',' INT_LITERAL)* ')';

MODEL:   'model';
GRAPH:   'graph';
CONFIG:  'config';
NODE:    'node';
EDGE:    'edge';
INPUT:   'input';
OUTPUT:  'output';
TENSOR:  'tensor';
LABEL:   'label';
NONE:    'None';

BOOL_LITERAL: 'true' | 'false';
ARROW:        '->';

FLOAT_LITERAL: [0-9]+ '.' [0-9]+;
INT_LITERAL:   [0-9]+;
STRING:        '"' (~["\r\n])* '"';
ID:            [a-zA-Z_][a-zA-Z0-9_]*;

WS:            [ \t]+    -> skip;
NEWLINE:       [\r\n]+   -> skip;
LINE_COMMENT:  '//' ~[\r\n]* -> skip;
BLOCK_COMMENT: '/*' .*? '*/' -> skip;
