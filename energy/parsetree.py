from math import log

class Node:
    def __init__(self, name):
        self.name = name
        self.isleaf = True
        self.children = None
        self.expand_p = 1

    def expand(self, children, expand_p):
        self.children = children
        self.expand_p = expand_p
        self.isleaf = False

    @staticmethod
    def recursive_log_p(topnode):
        if topnode.isleaf: return log(topnode.expand_p)
        else:
            ans = log(topnode.expand_p)
            for c in topnode.children:
                ans += Node.recursive_log_p(c)
            return ans


class ParseTree:
    def __init__(self, grammar, ast, log=None, expansions=None):
        self.grammar = grammar
        self.root = Node('A')
        if log: print("A => {}".format(ast.nt), file=log)
        if isinstance(expansions, list):
            expansions.append(("A", ast.nt))
        topnode = Node(ast.nt)
        self.recursive_parse(ast, topnode, log=log, expansions=expansions)
        self.root.expand([topnode], 1 / len(grammar.rules['A']))


    def recursive_parse(self, ast_node, topnode, lambda_stat="", prefix="", suffix="", log=None, expansions=None):
        nt = ast_node.nt
        if len(nt) == 1:
            nt = lambda_stat + nt
        if ast_node.ntype in {"lambda_x", "lambda_y"}:
            expansion = [ast_node.name]
        elif ast_node.name == "lambda":
            expansion = self.grammar.lookup[(ast_node.nt, ast_node.name, ast_node.nt[1:])]
        else:
            right_nt = None
            if ast_node.children:
                for c in ast_node.children:
                    if c.nt != "setLiteral":
                        right_nt = c.nt
                        break
                if right_nt and len(right_nt) == 1:
                    right_nt = lambda_stat + right_nt
            expansion = self.grammar.lookup[(nt, ast_node.name, right_nt)]

        children = []
        for x in expansion:
            children.append(Node(x))
        n_expansion_rules = len(self.grammar.rules[nt])
        topnode.expand(children, 1 / n_expansion_rules)
        if log:
            print("  =>{} {} {}".format(prefix, " ".join(expansion), suffix), file=log)
            #import pdb; pdb.set_trace()
        if isinstance(expansions, list):
            expansions.append((nt, expansion))

        current_suffix = ""
        if ast_node.children is not None:
            processed_children = 0
            if ast_node.ntype == 'lambda_op':    # skip the first child for lambda operation
                processed_children += 1

            for i, cnode in enumerate(topnode.children):
                if cnode.name in self.grammar.nonterminals:
                    if log: current_suffix = " ".join(expansion[i + 1:]) + " "
                    child_ast = ast_node.children[processed_children]
                    if ast_node.nt in {'fxB', 'fxN', 'fxL'}:
                        lambda_stat = "x"
                    elif ast_node.nt == 'fyL':
                        lambda_stat = "y"
                    self.recursive_parse(child_ast, cnode, lambda_stat,
                            prefix, current_suffix + suffix, log, expansions)
                    processed_children += 1
                    if log: prefix += " ( " + child_ast.prog + " )"
                else:
                    if log: prefix += " " + cnode.name


    def logp(self):
        return Node.recursive_log_p(self.root)