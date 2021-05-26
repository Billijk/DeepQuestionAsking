import torch
from torch import nn, optim
import torch.nn.functional as F

import os
import numpy as np
import argparse
import logging
from itertools import chain

import data
from data import DataLoader
from utils import cal_performance
from model import *

from torch.distributions import Categorical

parser = argparse.ArgumentParser()
parser.add_argument('data', type=str, default="../data", help="Path of dataset.")
parser.add_argument('save', type=str, default="../save", help="Path to save checkpoints.")
# training arguments
parser.add_argument('--seed', type=int, default=1, help="Random seed.")
parser.add_argument('--load', type=str, default="", help="Checkpoint to load.")
parser.add_argument('--epoch', type=int, default=1000, help="Max number of epochs to train.")
parser.add_argument('--save_epoch', type=int, default=50, help="Number of epochs per save point.")
parser.add_argument('--log', type=str, default="", help="Path to save log file.")
parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
parser.add_argument('--dropout', type=float, default=0.25, help="Dropout probability.")
# encoder arguments
parser.add_argument('--encode_size', type=int, default=50, help="Size of encoded vector. 32 as default.")
parser.add_argument('--channel', type=int, default=10, help="Number of output channels per Conv layer.")
parser.add_argument('--board_emb', type=str, default='fix_binary', help="Board embedding type. One of 'none', 'binary', 'fix_binary', 'rand'. 'none' as default.")
parser.add_argument('--board_emb_size', type=int, help="Board embedding size. Must be specified if board_emb is 'rand'.")
parser.add_argument('--pos_emb_size', type=int, default=0, help="Position embedding size. 0 (not used) as default.")
# decoder arguments
parser.add_argument('--emb_size', type=int, default=50, help="Word embedding size. 32 as default.")
parser.add_argument('--max_len', type=int, default=100, help="Maximum question length for each board.")
parser.add_argument('--hidden_size', type=int, default=50, help="Size of LSTM hidden states. 32 as default.")
parser.add_argument('--output_size', type=int, default=-1, help="Vocabulary size. -1 as default, determined by dataset.")
parser.add_argument('--n_head', type=int, default=8, help="Number of heads in multi-head attention.")
parser.add_argument('--n_layers', type=int, default=6, help="Number of layers in Transformer decoder.")
parser.add_argument('--d_k', type=int, default=64, help="d_k of attention.")
parser.add_argument('--d_v', type=int, default=64, help="d_v of attention.")
args = parser.parse_args()

torch.manual_seed(args.seed)

LOGGER_FORMAT = '[%(asctime)s] %(message)s'
logging.basicConfig(format=LOGGER_FORMAT, level=logging.INFO)
if len(args.log) > 0:
    logging.getLogger().addHandler(logging.FileHandler(args.log))


def main():
    train_loader = DataLoader(args.data, args.batch_size, None)
    if args.output_size == -1:
        args.output_size = len(train_loader.vocab)

    logging.info("Building models.")
    encoder = CNN_Encoder(args)
    decoder = Transformer_Decoder(args)
    logging.info(encoder)
    logging.info(decoder)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
        chain(encoder.parameters(), decoder.parameters())), lr=args.lr)
    encoder.train()
    decoder.train()
    encoder.cuda()
    decoder.cuda()

    # load checkpoint
    start_ep = 0
    if len(args.load) > 0:
        load_dict = torch.load(args.load)
        if 'epoch' in load_dict:
            start_ep = load_dict['epoch']
        if 'encoder' in load_dict:
            encoder.load_state_dict(load_dict['encoder'], strict=False)
        if 'decoder' in load_dict:
            decoder.load_state_dict(load_dict['decoder'], strict=False)
        if 'optimizer' in load_dict:
            try:
                optimizer.load_state_dict(load_dict['optimizer'])
            except:
                logging.warn("Cannot load optimizer parameters. New ones are used.")
        
    
    logging.info("Start Training.")
    for ep in range(start_ep, args.epoch):
        sum_loss, steps_cnt = 0, 0
        for batch in train_loader:
            # train step
            boards, targets, target_pos, *_ = batch
            boards, targets, target_pos = boards.cuda(), targets.cuda(), target_pos.cuda()
            board_feats = torch.normal(mean=torch.zeros((boards.size(0), 36, args.encode_size)), std=1).cuda()#encoder(boards)
            decoder_out, *_ = decoder(targets[:, :-1], target_pos[:, :-1], board_feats)
            loss, _ = cal_performance(decoder_out, targets[:, 1:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.data.cpu().item()
            steps_cnt += 1

        logging.info("Epoch {}: avg loss {:.4f}".format(ep, sum_loss / steps_cnt))

        if (ep + 1) % args.save_epoch == 0:
            save_dict = {
                'epoch': ep + 1,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_path = os.path.join(args.save, "ep_{}.pth".format(ep + 1))
            torch.save(save_dict, save_path)
            logging.info(" Checkpoint saved at {}".format(save_path))
    
if __name__ == "__main__":
    main()