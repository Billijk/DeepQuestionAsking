import torch
from torch import nn, optim
import torch.nn.functional as F

import os
import numpy as np
import argparse
import pickle
from itertools import chain
from copy import deepcopy

import json
import data
from model import *
from utils import cal_performance

parser = argparse.ArgumentParser()
parser.add_argument('data', type=str, help="Path of dataset.")
parser.add_argument('load', type=str, help="Checkpoint to load.")
parser.add_argument("output", type=str, help="Output path of the evaluation report.")
# training arguments
parser.add_argument('--eval-only', action="store_true", help="Do not train the model.")
parser.add_argument('--epoch', type=int, default=50, help="Number of epochs to train.")
parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
parser.add_argument('--dropout', type=float, default=0.25, help="Dropout probability.")
# encoder arguments
parser.add_argument("--no_encoder", action="store_true", help="Use decoder-only model.")
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
    prefix = ""
    board_paths = open(os.path.join(args.data, prefix + "human_board_files.txt")).read().splitlines()
    programs = json.load(open(os.path.join(args.data, prefix + "human_programs.json")))
    for board_file, program in zip(board_paths, programs):
        prog_tokens = []
        for p in program:
            p = p.replace("(", " ( ").replace(")", " ) ").split()
            prog_tokens.append([vocab.vocab[w] for w in p] + [data.SPECIAL_TOKENS['<eos>']])
        board = data.read_board(open(os.path.join(args.data, board_file)).read())
        eval_data.append((board, prog_tokens))
    return eval_data

def calc_probability(logits, target):
    #import pdb; pdb.set_trace()
    logprob = F.log_softmax(logits, dim=2).view(-1, logits.size(2))
    one_hot = torch.zeros_like(logprob).scatter(1, target.view(-1, 1), 1)
    sumlogprob = (one_hot * logprob).sum()
    return sumlogprob

def wrap_batch_data(board, targets):
    boards, texts, positions = [], [], []
    maxlen = 0
    for t in targets:
        boards.append(board)
        texts.append(t)
        maxlen = max(len(t), maxlen)

    # pad texts
    padded_texts = []
    for t in texts:
        padded_texts.append([data.SPECIAL_TOKENS['<start>']] + t + [data.SPECIAL_TOKENS['<pad>']] * (maxlen - len(t)))
        positions.append(list(range(0, len(t) + 1)) + [0] * (maxlen - len(t)))
    return (torch.LongTensor(boards).unsqueeze(1), torch.LongTensor(padded_texts), torch.LongTensor(positions))

def main():
    vocab = data.VocabLoader(args.data)
    if args.output_size == -1:
        args.output_size = len(vocab.vocab)
    eval_data = load_boards_targets(vocab)

    print("Building models.")
    encoder = CNN_Encoder(args)
    decoder = Transformer_Decoder(args)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
        chain(encoder.parameters(), decoder.parameters())), lr=args.lr)
    encoder.cuda()
    decoder.cuda()

    #load_dict = torch.load(args.load)
    load_dict = {
        'encoder': deepcopy(encoder.state_dict()),
        'decoder': deepcopy(decoder.state_dict())
    }
    if args.eval_only:
        encoder.load_state_dict(load_dict['encoder'])
        decoder.load_state_dict(load_dict['decoder'])

    print("Start cross validation:")
    perplexities = []
    reports_fp = open(args.output, "w")
    for val_id in range(16):
        # train part
        sum_loss = 0
        if not args.eval_only:
            encoder.load_state_dict(deepcopy(load_dict['encoder']))
            decoder.load_state_dict(deepcopy(load_dict['decoder']))
            encoder.train()
            decoder.train()
            for ep in range(args.epoch):
                for i, train_data in enumerate(eval_data):
                    if i == val_id:
                        continue

                    boards, targets, target_pos, *_ = wrap_batch_data(*train_data)
                    boards, targets, target_pos = boards.cuda(), targets.cuda(), target_pos.cuda()
                    if args.no_encoder:
                        board_feats = torch.zeros((boards.size(0), 36, args.encode_size)).cuda()
                    else:
                        board_feats = encoder(boards)
                    decoder_out, *_ = decoder(targets[:, :-1], target_pos[:, :-1], board_feats)
                    loss, _ = cal_performance(decoder_out, targets[:, 1:])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    sum_loss += (loss.data.cpu().item() / 15)

        # eval part
        perplex = 0
        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            board, targets = eval_data[val_id]
            input = torch.cuda.LongTensor(board).view(1, 1, 6, 6)
            board_feats = encoder(input)
            print(np.array(board), file=reports_fp)
            for target in targets:
                target_pos = torch.cuda.LongTensor(list(range(0, len(target) + 1))).view(1, -1)
                targets_tensor = torch.cuda.LongTensor([data.SPECIAL_TOKENS['<start>']] + target).view(1, -1)
                decoder_out, *_ = decoder(targets_tensor[:, :-1], target_pos[:, :-1], board_feats)
                logprobs = calc_probability(decoder_out, targets_tensor[:, 1:])
                perplex += logprobs.cpu().item()
                print("{:.4f} {}".format(logprobs.cpu().item(), vocab.translate(target)), file=reports_fp)
            print("", file=reports_fp)

        perplexities.append(perplex)
        #print(f"Iter {val_id}: loss {sum_loss / args.epoch:.4f}, loglikelihood {perplex}")
        print(f"Iter {val_id}: loss {sum_loss / args.epoch:.4f}, loglikelihood {perplex}", file=reports_fp)
    #print("")
    perplexities = np.array(perplexities)
    print("Average loglikelihood: {:.2f} high {:.2f} low {:.2f}".format(np.mean(perplexities), 
                np.mean(perplexities[[8, 1, 9, 0, 7]]), np.mean(perplexities[[15, 13, 12, 14, 6]])))
    print("Average loglikelihood: {:.2f}".format(np.mean(perplexities)), file=reports_fp)
    reports_fp.close()

if __name__ == "__main__":
    main()