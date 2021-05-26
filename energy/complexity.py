try:
    from .grammar import Grammar
    from .parsetree import ParseTree
except:
    from grammar import Grammar
    from parsetree import ParseTree
import sys
from eig.battleship import Parser
from eig.battleship.program import FUNC_NTYPES, DataType, LiteralNode, LambdaVarNode, ProgramSyntaxError

ntype_to_symbol = {k : v for v, k in FUNC_NTYPES.items()}
dtype_to_nonterminal = {
    DataType.NUMBER : 'N',
    DataType.BOOLEAN : 'B',
    DataType.COLOR : 'C',
    DataType.LOCATION : 'L',
    DataType.ORIENTATION : 'O',
    DataType.LAMBDA_X : 'C',
    DataType.LAMBDA_Y : 'L',
    DataType.LAMBDA_FXB : 'fxB',
    DataType.LAMBDA_FYB : 'fyB',
    DataType.LAMBDA_FXN : 'fxN',
    DataType.LAMBDA_FXL : 'fxL',
    DataType.SET_S : 'setS',
    DataType.SET_B : 'setB',
    DataType.SET_N : 'setN',
    DataType.SET_L : 'setL',
    DataType.SET_LITERAL_L : 'setLiteral',
    DataType.SET_LITERAL_S : 'setLiteral'
}

def rebuild_ast(node):
    # literal nodes need special processing
    if isinstance(node, LiteralNode) or isinstance(node, LambdaVarNode):
        node.name = node.prog
    else:
        node.name = ntype_to_symbol[node.ntype]
    node.nt = dtype_to_nonterminal[node.dtype]
    if node.children:
        for c in node.children:
            rebuild_ast(c)

class Complexity:
    def __init__(self, filename):
        self.grammar = Grammar(open(filename))
        #print("Terminals:", self.grammar.terminals)
        #print("Nonterminals:", self.grammar.nonterminals)
        #print("Rules:")
        #for left in self.grammar.rules:
        #    print("  {} -> {}".format(left, " | ".join([" ".join(x) for x in self.grammar.rules[left]])))

    def compute(self, prog, log=None):
        try:
            ast = Parser.parse(prog)
        except ProgramSyntaxError:
            return -1
        rebuild_ast(ast)
        #import pdb; pdb.set_trace()
        parse = ParseTree(self.grammar, ast, log=log)
        return -parse.logp()

if __name__ == "__main__":
    complexity = Complexity("grammar.txt")

    #print(complexity.compute("TRUE"))
    #print(complexity.compute("(colL 3-2)"))
    #print(complexity.compute("(and TRUE (== 5 2))"))
    print(complexity.compute("(> (++ (map (lambda x0 (== (size x0) 2)) (set AllColors))) 0)", log=sys.stdout))
