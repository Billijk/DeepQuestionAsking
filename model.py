import torch
from torch import nn, optim
import torch.nn.functional as F
import data
from utils import get_non_pad_mask, get_sinusoid_encoding_table, \
            get_subsequent_mask, get_attn_key_pad_mask
from modules import DecoderLayer

class CNN_Encoder(nn.Module):
    """
    Convolutional encoder:
        3 conv operations with kernel size 3*3, 4*4 and 5*5.
        All flattened and concatenated as outputs.
        No pooling; relu activation.
    """

    def __init__(self, opts):
        super().__init__()

        if opts.board_emb == 'none':
            input_size = 1
            self.use_board_emb = False
        elif opts.board_emb == 'binary' or opts.board_emb == 'fix_binary':
            input_size = 5
            self.use_board_emb = True
            self.board_emb = nn.Embedding.from_pretrained(torch.eye(5))
            if not opts.board_emb.startswith('fix_'):
                self.board_emb.weight.requires_grad = True
        elif opts.board_emb == 'rand':
            input_size = opts.board_emb_size
            self.use_board_emb = True
            self.board_emb = nn.Embedding(5, opts.board_emb_size)
        input_size += opts.pos_emb_size

        self.use_pos_emb = opts.pos_emb_size > 0
        if self.use_pos_emb:
            self.pos_emb = nn.Embedding(36, opts.pos_emb_size)

        self.conv1x1 = nn.Conv2d(input_size, opts.channel, 1, padding=0)
        self.conv3x3 = nn.Conv2d(input_size, opts.channel, 3, padding=1)
        self.conv5x5 = nn.Conv2d(input_size, opts.channel, 5, padding=2)

        self.channel = opts.channel
        self.fmap_full_size = 3 * opts.channel
        self.linear = nn.Linear(self.fmap_full_size, opts.encode_size)
        self.dropout = nn.Dropout(opts.dropout)

        # weight initialization
        nn.init.xavier_normal_(self.conv1x1.weight)
        nn.init.xavier_normal_(self.conv3x3.weight)
        nn.init.xavier_normal_(self.conv5x5.weight)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.conv1x1.bias, 0.01)
        nn.init.constant_(self.conv3x3.bias, 0.01)
        nn.init.constant_(self.conv5x5.bias, 0.01)
        nn.init.constant_(self.linear.bias, 0.01)

    def forward(self, inputs):
        """
        param inputs (Tensor batch x 1 x 6 x 6): battleship board
        ret (Tensor batch x encode_size): encoded board
            (Tensor batch x 3*channel x 36): CNN features of the board
        """
        # embedding
        if self.use_board_emb:
            inputs = self.board_emb(inputs - 1).transpose(1, 4).squeeze(4).contiguous()
        else:
            inputs = inputs.to(torch.float)
        if self.use_pos_emb:
            batch_size = inputs.size(0)
            positions = torch.arange(36, dtype=torch.long).cuda()
            position_embs = self.pos_emb(positions).transpose(0, 1).contiguous()
            position_embs = position_embs.view(1, -1, 6, 6).expand(batch_size, -1, -1, -1)
            inputs = torch.cat([inputs, position_embs], dim=1)

        x1 = torch.relu(self.conv1x1(inputs))
        x2 = torch.relu(self.conv3x3(inputs))
        x3 = torch.relu(self.conv5x5(inputs))
        cnn_feats = torch.cat([x.view(-1, self.channel, 36) for x in [x1, x2, x3]], dim=1)
        cnn_feats = self.dropout(cnn_feats.transpose(1, 2).contiguous())
        return self.linear(cnn_feats)

class Transformer_Decoder(nn.Module):
    def __init__(self, opts):

        super().__init__()
        n_position = opts.max_len + 1

        self.tgt_word_emb = nn.Embedding(opts.vocab_size, opts.emb_size, padding_idx=data.SPECIAL_TOKENS['<pad>'])
        self.tgt_word_prj = nn.Linear(opts.emb_size, opts.output_size, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, opts.emb_size, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(opts.emb_size, opts.hidden_size, opts.n_head, opts.d_k, opts.d_v, dropout=opts.dropout)
            for _ in range(opts.n_layers)])

    def forward(self, tgt_seq, tgt_pos, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        seq_logit = self.tgt_word_prj(dec_output)

        if return_attns:
            return seq_logit, dec_slf_attn_list, dec_enc_attn_list
        return seq_logit[:, 0],
