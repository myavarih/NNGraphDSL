# Generated from NNGraph.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .NNGraphParser import NNGraphParser
else:
    from NNGraphParser import NNGraphParser

# This class defines a complete listener for a parse tree produced by NNGraphParser.
class NNGraphListener(ParseTreeListener):

    # Enter a parse tree produced by NNGraphParser#start.
    def enterStart(self, ctx:NNGraphParser.StartContext):
        pass

    # Exit a parse tree produced by NNGraphParser#start.
    def exitStart(self, ctx:NNGraphParser.StartContext):
        pass


    # Enter a parse tree produced by NNGraphParser#model_block.
    def enterModel_block(self, ctx:NNGraphParser.Model_blockContext):
        pass

    # Exit a parse tree produced by NNGraphParser#model_block.
    def exitModel_block(self, ctx:NNGraphParser.Model_blockContext):
        pass


    # Enter a parse tree produced by NNGraphParser#input_decl.
    def enterInput_decl(self, ctx:NNGraphParser.Input_declContext):
        pass

    # Exit a parse tree produced by NNGraphParser#input_decl.
    def exitInput_decl(self, ctx:NNGraphParser.Input_declContext):
        pass


    # Enter a parse tree produced by NNGraphParser#output_decl.
    def enterOutput_decl(self, ctx:NNGraphParser.Output_declContext):
        pass

    # Exit a parse tree produced by NNGraphParser#output_decl.
    def exitOutput_decl(self, ctx:NNGraphParser.Output_declContext):
        pass


    # Enter a parse tree produced by NNGraphParser#graph_block.
    def enterGraph_block(self, ctx:NNGraphParser.Graph_blockContext):
        pass

    # Exit a parse tree produced by NNGraphParser#graph_block.
    def exitGraph_block(self, ctx:NNGraphParser.Graph_blockContext):
        pass


    # Enter a parse tree produced by NNGraphParser#node_decl.
    def enterNode_decl(self, ctx:NNGraphParser.Node_declContext):
        pass

    # Exit a parse tree produced by NNGraphParser#node_decl.
    def exitNode_decl(self, ctx:NNGraphParser.Node_declContext):
        pass


    # Enter a parse tree produced by NNGraphParser#layer_expr.
    def enterLayer_expr(self, ctx:NNGraphParser.Layer_exprContext):
        pass

    # Exit a parse tree produced by NNGraphParser#layer_expr.
    def exitLayer_expr(self, ctx:NNGraphParser.Layer_exprContext):
        pass


    # Enter a parse tree produced by NNGraphParser#param_list.
    def enterParam_list(self, ctx:NNGraphParser.Param_listContext):
        pass

    # Exit a parse tree produced by NNGraphParser#param_list.
    def exitParam_list(self, ctx:NNGraphParser.Param_listContext):
        pass


    # Enter a parse tree produced by NNGraphParser#param.
    def enterParam(self, ctx:NNGraphParser.ParamContext):
        pass

    # Exit a parse tree produced by NNGraphParser#param.
    def exitParam(self, ctx:NNGraphParser.ParamContext):
        pass


    # Enter a parse tree produced by NNGraphParser#edge_decl.
    def enterEdge_decl(self, ctx:NNGraphParser.Edge_declContext):
        pass

    # Exit a parse tree produced by NNGraphParser#edge_decl.
    def exitEdge_decl(self, ctx:NNGraphParser.Edge_declContext):
        pass


    # Enter a parse tree produced by NNGraphParser#config_block.
    def enterConfig_block(self, ctx:NNGraphParser.Config_blockContext):
        pass

    # Exit a parse tree produced by NNGraphParser#config_block.
    def exitConfig_block(self, ctx:NNGraphParser.Config_blockContext):
        pass


    # Enter a parse tree produced by NNGraphParser#config_entry.
    def enterConfig_entry(self, ctx:NNGraphParser.Config_entryContext):
        pass

    # Exit a parse tree produced by NNGraphParser#config_entry.
    def exitConfig_entry(self, ctx:NNGraphParser.Config_entryContext):
        pass


    # Enter a parse tree produced by NNGraphParser#value.
    def enterValue(self, ctx:NNGraphParser.ValueContext):
        pass

    # Exit a parse tree produced by NNGraphParser#value.
    def exitValue(self, ctx:NNGraphParser.ValueContext):
        pass


    # Enter a parse tree produced by NNGraphParser#shape_expr.
    def enterShape_expr(self, ctx:NNGraphParser.Shape_exprContext):
        pass

    # Exit a parse tree produced by NNGraphParser#shape_expr.
    def exitShape_expr(self, ctx:NNGraphParser.Shape_exprContext):
        pass



del NNGraphParser