import torch
import torch.nn.functional as F
import numpy as np
from math import log2

import os, sys
import eig
from eig.battleship import BattleshipHypothesisSpace, Parser, Executor
from eig.battleship.program import ProgramSyntaxError
from energy.energy import infer, EnergyModel
from energy.complexity import Complexity
from energy.data import append_nonboard_features

import data

class Reward:
    def __init__(self, opts):
        self.reward = opts.reward
        self.hypotheses = BattleshipHypothesisSpace(grid_size=6, ship_labels=[1, 2, 3], ship_sizes=[2, 3, 4], orientations=['V', 'H'])
        if self.reward == "energy":
            self.complexity = Complexity(os.path.join(opts.energy_path, "grammar.txt"))
            bag_nb_feats = torch.load(os.path.join(opts.energy_path, "bagfeats/non_board.tensor"))
            bag_feats = []
            for i in range(18):
                eig_feats = torch.load(os.path.join(opts.energy_path, "bagfeats/eig_board_{}.tensor".format(i + 1)))
                bag_feats.append(torch.cat([eig_feats, bag_nb_feats], dim=1))
            bag_feats = torch.cat(bag_feats, dim=0)
            self.minv, _ = bag_feats.min(dim=0)
            self.maxv, _ = bag_feats.max(dim=0)
            self.energy_model = EnergyModel()
            self.energy_model.load_state_dict(torch.load(os.path.join(opts.energy_path, 
                        "weights2.pth"), map_location=lambda storage, loc: storage))


    def prepare_infer_nonboard(self, progs_ref):
        nonboard_feats = []
        for i, prog in enumerate(progs_ref):
            prog_feats = []
            try:
                append_nonboard_features(prog, prog_feats, self.complexity)
            except:
                import traceback
                traceback.print_exc()
                print("Error on program:", prog)
                sys.exit(-1)
            nonboard_feats.append(prog_feats)
        nonboard_feats = torch.Tensor(nonboard_feats)
        return nonboard_feats


    def compute_energy(self, prog, eig, correctness):
        if correctness == 0: return 0
        else:
            nonboard_feats = self.prepare_infer_nonboard([prog])
            eigs = torch.Tensor([eig]).view(-1, 1)
            eigs_01 = (eigs <= 1e-6).to(torch.float32)
            feats = (torch.cat([eigs, eigs_01, nonboard_feats], dim=1) - self.minv) / (self.maxv - self.minv)
            weights = infer(feats, self.energy_model)
            return weights[0]


    def compute_eig(self, program, valid_ids):
        correctness = 1
        try:
            question = Parser.parse(program)
            executor = Executor(question)
            answers = self.hypotheses.execute_on_subspace(executor, valid_ids)
            single_prob = 1 / len(valid_ids)
            answer_probs = {}
            for ans in answers:
                if not ans in answer_probs:
                    answer_probs[ans] = 0.
                answer_probs[ans] += single_prob
            score = (- sum([p * log2(p) for p in answer_probs.values()]))
        except (ProgramSyntaxError, RuntimeError):
            score = -1
            correctness = 0
        except:
            import traceback
            traceback.print_exc()
            print("Error on program:", program)
            sys.exit(-1)
        return score, correctness

    
    def get_reward(self, program, context):
        e, c = self.compute_eig(program, context)
        if self.reward == 'eig':
            return -1 if c == 0 else e / 4
        elif self.reward == 'energy':
            ev = self.compute_energy(program, e, c)
            reward = -2 if c == 0 else -ev / 10
            return min(max(reward, -2), 1) + 1
    

def sequence_mask(sequence_length, max_len=None):
    """
    From https://github.com/howardyclo/pytorch-seq2seq-example/
    """
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    mask = seq_range_expand < seq_length_expand
    
    return mask

def masked_cross_entropy(logits, target, length):
    """
    From https://github.com/howardyclo/pytorch-seq2seq-example/
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    # Note: mask need to bed casted to float!
    losses = losses * mask.float()
    loss = losses.sum() / mask.float().sum()
    
    # (batch_size * max_tgt_len,)
    pred_flat = log_probs_flat.max(1)[1]
    # (batch_size * max_tgt_len,) => (batch_size, max_tgt_len) => (max_tgt_len, batch_size)
    pred_seqs = pred_flat.view(*target.size()).transpose(0,1).contiguous()
    # (batch_size, max_len) => (batch_size * max_tgt_len,)
    mask_flat = mask.view(-1)
    
    # `.float()` IS VERY IMPORTANT !!!
    # https://discuss.pytorch.org/t/batch-size-and-validation-accuracy/4066/3
    num_corrects = int(pred_flat.eq(target_flat.squeeze(1)).masked_select(mask_flat).float().data.sum())
    num_words = length.data.sum()

    return loss, pred_seqs, num_corrects, num_words

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(data.SPECIAL_TOKENS['<pad>']).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(data.SPECIAL_TOKENS['<pad>'])
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask