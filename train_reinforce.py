import torch
from torch import nn, optim
import torch.nn.functional as F

import sys
import os

import math
import numpy as np
import argparse
import logging
from itertools import chain

import data
from data import BoardLoader
from env import Env
from model import *

from torch.distributions import Categorical

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('data', type=str, default="../data/bigbag", help="Path of dataset.")
parser.add_argument('save', type=str, default="../save", help="Path to save checkpoints.")
parser.add_argument('--energy_path', type=str, default="./energy/", help="Path of energy model.")
# training arguments
parser.add_argument('--reward', type=str, default='energy', help="Reward function to use. [eig|energy]")
parser.add_argument('--step_penalty', type=float, default='0.01', help="Step penalty of each step.")
parser.add_argument('--seed', type=int, default=1, help="Random seed.")
parser.add_argument('--load', type=str, default="", help="Checkpoint to load.")
parser.add_argument('--epoch', type=int, default=1000, help="Max number of epochs to train.")
parser.add_argument('--gamma', type=float, default=0.99, help="Temporal discount factor for calculating step rewards.")
parser.add_argument('--save_epoch', type=int, default=50, help="Number of epochs per save point.")
parser.add_argument('--log', type=str, default="", help="Path to save log file.")
parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
parser.add_argument('--dropout', type=float, default=0.25, help="Dropout probability.")
parser.add_argument('--creativity', action="store_true", help="Reward for novel questions.")
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
#parser.add_argument('--output_size', type=int, default=-1, help="Vocabulary size. -1 as default, determined by dataset.")
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
    vocab = data.VocabLoader(args.data)
    env = Env(args, vocab)
    train_loader = BoardLoader(args.data, args.batch_size, env.reward.hypotheses)
    args.vocab_size = len(vocab)
    args.output_size = len(env.rewrites)

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

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    logging.info("Start Training.")
    for ep in range(start_ep, args.epoch):
        sum_loss, sum_reward, batch_cnt, max_reward = 0, 0, 0, -1
        rand_ratio = max(0.01,  math.exp((-50 - ep) / 70))
        #rand_ratio = 0.01
        for boards, contexts in train_loader:
            optimizer.zero_grad()

            # train step
            boards = boards.cuda()
            board_feats = encoder(boards)
            log_probs, rewards, masks = [], [], []

            states = env.reset(len(contexts))
            done_last = np.copy(env.done)
            step = 0
            local_rand_ratio = rand_ratio
            while True:
                step += 1
                if step == 10:
                    local_rand_ratio = local_rand_ratio / 2
                elif step > 15:
                    local_rand_ratio = 0
                inputs, positions = states
                action_logits, *_ = decoder(inputs, positions, board_feats)
                # sample an action
                try:
                    action_masks = env.legal_actions()
                    #raw_probs = F.softmax(action_logits, dim=1)
                    legal_action_logits = torch.ones_like(action_logits) * -1e10
                    for i, m in enumerate(action_masks):
                        legal_action_logits[i, m] = action_logits[i, m]
                    dist = Categorical(logits=legal_action_logits)
                    rand = np.random.rand()
                    if rand < local_rand_ratio:
                        actions = env.sample()
                        action_tensor = torch.cuda.LongTensor(actions)
                    else:
                        action_tensor = dist.sample()
                        actions = action_tensor.cpu().numpy()
                    log_probs.append(dist.log_prob(action_tensor))
                    # update the environment
                    states, step_rewards, done = env.step(actions, contexts)

                except:
                    import traceback
                    traceback.print_exc()
                    logging.error(env.states)
                    logging.error(action_masks)
                    logging.error(actions)
                    sys.exit(-1)
                rewards.append(step_rewards)
                masks.append(done & done_last)
                done_last = np.copy(done)
                if np.all(done): break

            # finish episode
            rewards = torch.Tensor(rewards).cuda()
            returns = torch.zeros_like(rewards).cuda()
            returns[-1] = rewards[-1]
            for i in range(rewards.size(0) - 2, -1, -1):
                returns[i] = rewards[i] + args.gamma * returns[i + 1]
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
            masks = 1 - torch.FloatTensor(masks).cuda()

            policy_loss = []
            for log_prob, R, mask in zip(log_probs, returns, masks):
                policy_loss.append(-log_prob * R * mask)
            optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()

            sum_loss += policy_loss.data.cpu().item()
            sum_reward += rewards.cpu().sum().item() / len(boards)
            max_reward = max(max_reward, rewards.cpu().max().item())
            batch_cnt += 1

        mean_reward = sum_reward / batch_cnt
        logging.info("Epoch {} (rand {:.2f}): avg loss {:.4f} avg reward {:.4f} max reward {:.4f}".format(
            ep, rand_ratio, sum_loss / batch_cnt, mean_reward, max_reward))

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

            # also run validation on part of training set to get a bag of sampled programs
            if args.creativity:
                validation_programs = set()
                with torch.no_grad():
                    for i, batch in enumerate(train_loader):
                        if i == 50: break   # 50 batches
                        boards, _ = batch
                        boards = boards.cuda()
                        board_feats = encoder(boards)

                        states = env.reset(boards.size(0))
                        while True:
                            inputs, positions = states
                            action_logits, *_ = decoder(inputs, positions, board_feats)
                            action_masks = env.legal_actions()
                            legal_action_logits = torch.ones_like(action_logits) * -1e10
                            for j, m in enumerate(action_masks):
                                legal_action_logits[j, m] = action_logits[j, m]
                            dist = Categorical(logits=legal_action_logits)
                            action_tensor = dist.sample()
                            actions = action_tensor.cpu().numpy()
                            states, _, done = env.step(actions, None, train=False)
                            if np.all(done): break

                        validation_programs.update(env.sentences)
                logging.info(" {} programs are sampled on validation set".format(len(validation_programs)))
                env.prog_set = validation_programs


if __name__ == "__main__":
    main()
