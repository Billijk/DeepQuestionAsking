import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical

import os
import numpy as np
import argparse
import pickle
from itertools import chain
from eig import compute_eig_fast
from eig.battleship.program import ProgramSyntaxError

import data
from model import *
from env import Env

parser = argparse.ArgumentParser()
parser.add_argument('data', type=str, help="Path of dataset.")
parser.add_argument('load', type=str, help="Checkpoint to load.")
parser.add_argument('output', type=str, help="Path to save output sentences.")
parser.add_argument('--energy_path', type=str, default="./energy/", help="Path of energy model.")
# encoder arguments
parser.add_argument('--encode_size', type=int, default=50, help="Size of encoded vector. 32 as default.")
parser.add_argument('--channel', type=int, default=10, help="Number of output channels per Conv layer.")
parser.add_argument('--board_emb', type=str, default='fix_binary', help="Board embedding type. One of 'none', 'binary', 'fix_binary', 'rand'. 'none' as default.")
parser.add_argument('--board_emb_size', type=int, help="Board embedding size. Must be specified if board_emb is 'rand'.")
parser.add_argument('--pos_emb_size', type=int, default=0, help="Position embedding size. 0 (not used) as default.")
# decoder arguments
parser.add_argument('--emb_size', type=int, default=50, help="Word embedding size. 32 as default.")
parser.add_argument('--hidden_size', type=int, default=50, help="Size of LSTM hidden states. 32 as default.")
parser.add_argument('--output_size', type=int, default=-1, help="Vocabulary size. -1 as default, determined by dataset.")
parser.add_argument('--max_len', type=int, default=100, help="Maximum question length for each board.")
parser.add_argument('--n_head', type=int, default=8, help="Number of heads in multi-head attention.")
parser.add_argument('--n_layers', type=int, default=6, help="Number of layers in Transformer decoder.")
parser.add_argument('--d_k', type=int, default=64, help="d_k of attention.")
parser.add_argument('--d_v', type=int, default=64, help="d_k of attention.")
args = parser.parse_args()

def load_boards():
    paths = os.path.join(args.data, "test_board_files.txt")
    img_files = open(paths).read().splitlines()
    imgs = []
    for fname in img_files:
        full_path = os.path.join(args.data, fname)
        imgs.append(data.read_board(open(full_path).read()))
    return imgs

def calc_eig(board, prog):
    hs = {
        "grid_size": 6,
        "ship_labels": [1, 2, 3],
        "ship_sizes": [2, 3, 4],
        "orientations": ['V', 'H']
    }
    try:
        board = np.array(board) - 2
        return compute_eig_fast(prog, board, **hs)
    except (ProgramSyntaxError, RuntimeError):
        return 0

def load_train_progs():
    programs = open(os.path.join(args.data, "train_programs.txt")).read().splitlines()
    return set([" ".join(p.replace("(", " ( ").replace(")", " ) ").split()) for p in programs])

def main():
    vocab = data.VocabLoader(args.data)
    env = Env(args, vocab, train=False)
    args.vocab_size = len(vocab)
    args.output_size = len(env.rewrites)
    boards = load_boards()

    print("Building models.")
    args.dropout = 0
    encoder = CNN_Encoder(args)
    decoder = Transformer_Decoder(args)
    encoder.eval()
    decoder.eval()
    encoder.cuda()
    decoder.cuda()

    load_dict = torch.load(args.load)
    encoder.load_state_dict(load_dict['encoder'])
    decoder.load_state_dict(load_dict['decoder'])

    print("Start generating questions.")
    eigs = []
    unique_progs = set()
    out_f = open(args.output, 'w')
    for i, board in enumerate(boards):
        input = torch.cuda.LongTensor(board).view(1, 1, 6, 6)
        board_feats = encoder(input)

        #states = env.set_state(["B"])
        states = env.reset(1)
        while True:
            inputs, positions = states
            action_logits, *_ = decoder(inputs, positions, board_feats)
            # sample an action
            action_masks = env.legal_actions()
            legal_action_logits = torch.ones_like(action_logits) * -1e10
            for j, m in enumerate(action_masks):
                legal_action_logits[j, m] = action_logits[j, m]
            dist = Categorical(logits=legal_action_logits)
            action_tensor = dist.sample()
            actions = action_tensor.cpu().numpy()
            #actions = torch.argmax(legal_action_probs, dim=1).cpu().numpy()
            # update the environment
            states, _, done = env.step(actions, None)
            if np.all(done): break

        program = env.sentences[0]
        eigs.append(calc_eig(board, program))
        unique_progs.add(program)
        print(program, file=out_f)
        #print("\r Finished {} / {}".format(i + 1, len(boards)), end='', flush=True)
    print("")
    print("Inference finished. Results saved to {}".format(args.output))
    out_f.close()

    # metrics: avg_eig / %eig>3 / %eig>0 / %eig>0.95 / #unique / #unique_novel
    print("")
    print("=" * 20)
    print("Avg eig: {}".format(sum(eigs) / len(eigs)))
    print("=" * 20)
    eigs = sorted(eigs, reverse=True)
    i, n = 0, len(eigs)
    for t in [3, 0.95, 1e-4]:
        while (i < n and eigs[i] > t): i += 1
        print("{:.2f}% ({} / {}) programs have eigs greater than {}".format(i / n * 100, i, n, t))
    print("=" * 20)
    print("Unique programs: {}".format(len(unique_progs)))
    train_progs = load_train_progs()
    num_novel = 0
    for p in unique_progs:
        if p not in train_progs:
            num_novel += 1
    print(f"Unique programs not in training set: {num_novel}")
    print("=" * 20)

if __name__ == "__main__":
    main()