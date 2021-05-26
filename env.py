import os
import random
from io import StringIO

import torch
import numpy as np
from eig.battleship import Parser

from utils import Reward
from energy.grammar import Grammar
from energy.parsetree import ParseTree
from energy.complexity import ntype_to_symbol, dtype_to_nonterminal, rebuild_ast

class Env:
    def __init__(self, opts, vocab, train=True):
        self.grammar = Grammar(open(os.path.join(opts.energy_path, "grammar.txt")))
        self.train = train
        if train:
            self.reward = Reward(opts)
            self.step_penalty = opts.step_penalty
        self.prog_set = None
        self.vocab = vocab
        self.max_len = opts.max_len
        self.pad = vocab['<pad>']
        self.start_symbol = vocab['A']
        self.states = None
        self.sentences = None
        self.next_nt = None
        self.done = None
        self.nt, self.rewrites, self.rewrite_ids = self._tokenize_grammar(self.grammar, self.vocab)


    def reset(self, batch_size):
        # equals self.set_state(["A"] * batch_size), but a little faster
        self.states = [[self.start_symbol] for _ in range(batch_size)]
        self.next_nt = [self.start_symbol] * batch_size
        self.sentences = [None] * batch_size
        self.done = np.zeros([batch_size], dtype=bool)
        pad = torch.ones([batch_size, self.max_len - 1], dtype=torch.long) * self.pad
        inputs = torch.cat((torch.cuda.LongTensor(self.states), pad.cuda()), dim=1)
        positions = torch.zeros((batch_size, self.max_len), dtype=torch.long).cuda()
        states = (inputs, positions)
        return states

    
    def set_state(self, states):
        self.states, self.next_nt = [], []
        for state in states:
            next_nt = None
            new_string = []
            for t in state.split():
                x = self.vocab[t]
                if x in self.nt:    # non-terminal found, expand by randomness
                    if next_nt is None: next_nt = x
                new_string.append(x)
            self.states.append(new_string)
            self.next_nt.append(next_nt)
        self.sentences = [None] * len(states)
        self.done = np.zeros([len(states)], dtype=bool)
        positions = [list(range(0, len(x))) + [0] * (self.max_len - len(x)) for x in self.states]
        inputs = [x + [self.pad] * (self.max_len - len(x)) for x in self.states]
        states = (torch.cuda.LongTensor(inputs), torch.cuda.LongTensor(positions))
        return states

    
    def random_reset(self, batch_size):
        self.states, self.next_nt = [], []
        for _ in range(batch_size):
            state = [self.start_symbol]
            next_nt = self.start_symbol
            for level in range(5):         # expand 5 levels
                status = 1
                next_nt = None
                new_string = []
                for t in state:
                    if t in self.nt:    # non-terminal found, expand by randomness
                        if random.randint(0, 4) >= level:
                            production = self.rewrites[random.choice(self.rewrite_ids[t])]
                            new_string.extend(production[1])
                            for x in production[1]:
                                if x in self.nt:
                                    if next_nt is None: next_nt = x
                            applied = True
                            continue
                        else:           # more than one non-terminal
                            if next_nt is None: next_nt = t
                    new_string.append(t)
                if (next_nt is None) or (len(new_string) > self.max_len):
                    state = [self.start_symbol]
                    next_nt = self.start_symbol
                else:
                    state = new_string
            self.states.append(state)
            self.next_nt.append(next_nt)
        self.sentences = [None] * batch_size
        self.done = np.zeros([batch_size], dtype=bool)
        positions = [list(range(0, len(x))) + [0] * (self.max_len - len(x)) for x in self.states]
        inputs = [x + [self.pad] * (self.max_len - len(x)) for x in self.states]
        states = (torch.cuda.LongTensor(inputs), torch.cuda.LongTensor(positions))
        return states


    def step(self, actions, contexts, train=None):
        """
        return states, rewards, done
        """
        rewards = []
        if train is None: train = self.train
        for i in range(len(actions)):
            if not self.done[i]:
                state = self.states[i]
                a = actions[i]
                new_state, status, next_nt = self.one_step_derivation(state, self.rewrites[a])
                if next_nt is None: next_nt = self.start_symbol
                self.next_nt[i] = next_nt

                if (status == 0) and (len(new_state) <= self.max_len):
                    self.states[i] = new_state
                    if train:
                        rewards.append(-self.step_penalty)       # step penalty?
                else:
                    sentence = self.vocab.translate(new_state)
                    self.states[i] = [self.start_symbol]
                    self.sentences[i] = sentence
                    self.done[i] = True
                    if not train: continue
                    if status == -1:
                        rewards.append(-1)
                    else:
                        r = self.reward.get_reward(sentence, contexts[i])
                        if self.prog_set and (sentence in self.prog_set):
                            r -= 0.25
                        rewards.append(r)
            else:
                rewards.append(0)
        positions = [list(range(0, len(x))) + [0] * (self.max_len - len(x)) for x in self.states]
        inputs = [x + [self.pad] * (self.max_len - len(x)) for x in self.states]
        states = (torch.cuda.LongTensor(inputs), torch.cuda.LongTensor(positions))
        return states, rewards, self.done


    def legal_actions(self):
        actions = []
        for n in self.next_nt:
            actions.append(self.rewrite_ids[n])
        return actions

    def sample(self):
        actions = []
        for n in self.next_nt:
            actions.append(random.choice(self.rewrite_ids[n]))
        return actions


    def one_step_derivation(self, string, production):
        """
        Using left-most derivation
        Scan from left to right, find the left-most non-terminal, and try to apply the production. 
        Return the new string, and a status (0:normal, 1:done, -1:error)
        """
        status = 1
        next_nt = None
        applied = False
        new_string = []
        for t in string:
            if t in self.nt:    # non-terminal found
                if not applied: # try to apply the production
                    if t == production[0]:
                        new_string.extend(production[1])
                        for x in production[1]:
                            if x in self.nt:
                                if next_nt is None:
                                    next_nt = x
                                status = 0
                        applied = True
                        continue
                    else:       # fail to apply the production
                        return string, -1, None
                else:           # more than one non-terminal
                    if next_nt is None:
                        next_nt = t
                    status = 0
            new_string.append(t)
        return new_string, status, next_nt

    
    def get_action_seq(self, string):
        """
        Get a sequence of actions that can produce the given string from start symbol.
        Return a list of intermediate states and a list of action ids.
        """
        states = ["A"]
        actions = []
        ast = Parser.parse(string)
        rebuild_ast(ast)
        io_handler = StringIO()
        expansions = []
        __ = ParseTree(self.grammar, ast, log=io_handler, expansions=expansions)
        for expansion, state in zip(expansions, io_handler.getvalue().splitlines()):
            #if len(states) == 5:
            #    import pdb; pdb.set_trace()
            states.append(state.split("=>")[-1].strip())
            e_left = self.vocab[expansion[0]]
            e_right = [self.vocab[t] for t in expansion[1]]
            for i in self.rewrite_ids[e_left]:
                right = self.rewrites[i][1]
                if right == e_right:
                    actions.append(i)
                    break
        return states[:-1], actions

    def _tokenize_grammar(self, grammar, vocab):
        """
        tokenize non-terminals and rewrite rules
        """
        nt, rewrites, sample_ids = set(), [], {}
        for x in grammar.nonterminals:
            nt.add(vocab[x])
        for x in grammar.rules:
            id_list = []
            for y in grammar.rules[x]:
                right = []
                for t in y: right.append(vocab[t])
                id_list.append(len(rewrites))
                rewrites.append((vocab[x], right))
            sample_ids[vocab[x]] = id_list
        return nt, rewrites, sample_ids


# unittest
if __name__ == "__main__":
    import argparse
    from data import VocabLoader

    opts = argparse.Namespace()
    opts.energy_path = "./energy/"
    opts.reward = "eig"
    opts.max_len = 100
    opts.step_penalty = 0

    vocab_path = "../data/exp3"
    vocab = VocabLoader(vocab_path)
    contexts = [[0, 1, 2, 3, 4, 5, 6, 7]]
    env = Env(opts, vocab)
    
    #string = "(colL 3-2)"
    string = "(> (++ (map (lambda x0 (== (size x0) 2)) (set AllColors))) 0)"
    states, actions = env.get_action_seq(string)
    for s, a in zip(states, actions):
        left, right = env.rewrites[a]
        print(f"State {s}, Action {a} ({vocab.translate([left])}, {vocab.translate(right)}) => ")
    print(string)
    """
    # 1. invalid actions
    env.reset(1)
    s, r, d = env.step([10], contexts)

    # 2. valid actions
    env.reset(1)
    s, r, d = env.step([0], contexts)
    s, r, d = env.step([7], contexts)
    s, r, d = env.step([5], contexts)

    env.random_reset(10)[0]
    for s in env.states:
        print(env.vocab.translate(s))
    """