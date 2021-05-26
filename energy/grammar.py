import sys

class Grammar:
    def __init__(self, fp):
        self.nonterminals = set()
        self.terminals = set()
        self.rules = {}
        
        error_flag = False
        for i, rawline in enumerate(fp.readlines()):
            # first remove comments
            line = rawline[:rawline.find('#')]
            line = line.strip()
            if len(line) == 0: continue
            if line.startswith('"'):
                # non terminals
                if len(self.rules) > 0:
                    print("[ERROR] (Line {}) Nonterminals declarations should appear before rules. Ignored".format(i + 1))
                    print("        {}".format(rawline))
                    error_flag = True
                ts = line.split()
                for t in ts:
                    self.nonterminals.add(t[1:-1])
            elif line.count('::') == 1:
                # rules
                ts = line.split('::')
                left = ts[0].strip()
                if left not in self.nonterminals:
                    print("[ERROR] (Line {}) {} is not a nonterminal.".format(i + 1, left))
                    print("        {}".format(rawline))
                    error_flag = True
                
                right = ts[1].split()
                for t in right:
                    if t not in self.nonterminals:
                        self.terminals.add(t)

                if left not in self.rules:
                    self.rules[left] = []
                self.rules[left].append(right)                
            else:
                print("[WARNING] (Line {}) Unrecognizable. Ignored.".format(i + 1))
                print("          {}".format(rawline))
        
        if error_flag: sys.exit(-1)
        nrule = sum([len(self.rules[x]) for x in self.rules])
        print("{} nonterminals, {} terminals and {} rules are found.".format(len(self.nonterminals), len(self.terminals), nrule))

        # lookup of (nonterminal, operator symbol) -> expansion rule
        self.lookup = {}
        for left in self.rules:
            for right in self.rules[left]:
                # TODO: be careful about lambdas
                if right[0] == '(':
                    op = right[1]
                else: op = right[0]
                right_nt = None
                for token in right:
                    if token in self.nonterminals:
                        right_nt = token
                        break
                self.lookup[(left, op, right_nt)] = right