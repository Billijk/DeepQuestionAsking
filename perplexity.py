import torch
from torch import nn, optim
import torch.nn.functional as F

import os
import numpy as np
import argparse
import pickle
from itertools import chain

import data
from model import *
from train import cal_performance

parser = argparse.ArgumentParser()
parser.add_argument('data', type=str, help="Path of dataset.")
parser.add_argument('load', type=str, help="Checkpoint to load.")
# encoder arguments
parser.add_argument('--encode_size', type=int, default=50, help="Size of encoded vector. 32 as default.")
parser.add_argument('--channel', type=int, default=10, help="Number of output channels per Conv layer.")
parser.add_argument('--board_emb', type=str, default='none', help="Board embedding type. One of 'none', 'binary', 'fix_binary', 'rand'. 'none' as default.")
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

def load_boards_targets(vocab):
    eval_data = []
    board_paths = open(os.path.join(args.data, "test_board_files.txt")).read().splitlines()
    programs = open(os.path.join(args.data, "test_programs.txt")).read().splitlines()
    for board_file, program in zip(board_paths, programs):
        p = program.replace("(", " ( ").replace(")", " ) ").split()
        board = data.read_board(open(os.path.join(args.data, board_file)).read())
        prog_tokens = [vocab.vocab[w] for w in p] + [data.SPECIAL_TOKENS['<eos>']]
        eval_data.append((board, prog_tokens))
    return eval_data

def main():
    vocab = data.VocabLoader(args.data)
    if args.output_size == -1:
        args.output_size = len(vocab.vocab)
    eval_data = load_boards_targets(vocab)

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

    print("Start evaluating on test set.")
    perplexities = []
    for i, (input, targets) in enumerate(eval_data):
        input = torch.cuda.LongTensor(input).view(1, 1, 6, 6)
        target_pos = torch.cuda.LongTensor(list(range(0, len(targets) + 1))).view(1, -1)
        targets = torch.cuda.LongTensor([data.SPECIAL_TOKENS['<start>']] + targets).view(1, -1)
        board_feats_compress, _ = encoder(input)
        decoder_out, *_ = decoder(targets[:, :-1], target_pos[:, :-1], board_feats_compress.unsqueeze(1))
        loss, _ = cal_performance(decoder_out, targets[:, 1:])
        perplexities.append(loss.exp().cpu().item())
        print("\r Finished {} / {}".format(i + 1, len(eval_data)), end='', flush=True)
    print("")
    print("Total perplexity:", sum(perplexities) / len(perplexities))

    

if __name__ == "__main__":
    main()