import os
import random
import pickle
import copy
import logging
import json

import eig
import numpy as np
import torch

BOARD_TO_ID = {'H': 1, 'W': 2, 'B': 3, 'R': 4, 'P': 5}
SPECIAL_TOKENS = {'<start>': 0, '<eos>': 1, '<pad>': 2}

def read_board(board_raw):
    board = []
    for row in board_raw.splitlines():
        board.append([BOARD_TO_ID[x] for x in row.split(',')])
    return board

class VocabLoader(object):
    def __init__(self, data_path):
        vocab_file = os.path.join(data_path, "..", "prog_vocab.pkl")
        if not os.path.exists(vocab_file):
            # process one
            prog_tokens_file = os.path.join(data_path, "..", "prog_tokens.txt")
            tokens = open(prog_tokens_file).read().splitlines()
            vocab = copy.deepcopy(SPECIAL_TOKENS)
            for t in tokens: vocab[t] = len(vocab)
            pickle.dump(vocab, open(vocab_file, 'wb'))
            self.vocab = vocab
        else:
            self.vocab = pickle.load(open(vocab_file, 'rb'))
        self.ivocab = {v : k for k, v in self.vocab.items()}

    def translate(self, token_ids):
        """
        Translate from a list of ids to a string of tokens
        """
        tokens = []
        for id in token_ids:
            if id == SPECIAL_TOKENS['<eos>']: break
            tokens.append(self.ivocab[id])
        return ' '.join(tokens)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.ivocab[key]
        else:
            return self.vocab[key]

    def __len__(self):
        return len(self.vocab)


class DataLoader(object):
    def __init__(self, data_path, batch_size, hypotheses, drop_last=False):
        self.data, self.vocab = self._process_data(data_path, hypotheses)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def _wrap_data(self, batch_data):
        # wrap data with numpy array
        boards, texts, positions, lengths, contexts = [], [], [], [], []
        maxlen = 0
        for item in batch_data:
            b = item[0]
            t = random.choice(item[1])
            if len(item) == 3:
                contexts.append(item[2])
            boards.append(b)
            texts.append(t)
            lengths.append(len(t))
        maxlen = max(lengths)
        # pad texts
        padded_texts = []
        for t in texts:
            padded_texts.append([SPECIAL_TOKENS['<start>']] + t + [SPECIAL_TOKENS['<pad>']] * (maxlen - len(t)))
            positions.append(list(range(0, len(t) + 1)) + [0] * (maxlen - len(t)))
        return torch.LongTensor(boards).unsqueeze(1), torch.LongTensor(padded_texts), torch.LongTensor(positions), \
            torch.LongTensor(lengths), contexts

    def __iter__(self):
        """
        return (FloatTensor (batch_size x 1 x 6 x 6), LongTensor(batch_size x max_len), LongTensor(batch_size)):
            batch data of board, questions, lengths of questions
        """
        random.shuffle(self.data)
        batch_data = []
        for item in self.data:
            batch_data.append(item)
            if len(batch_data) == self.batch_size:
                yield self._wrap_data(batch_data)
                batch_data = []
        if (not self.drop_last) and (len(batch_data) > 0):
            yield self._wrap_data(batch_data)

    def __len__(self):
        if self.drop_last:
            return len(self.data) // self.batch_size
        else:
            return len(self.data  + self.batch_size - 1) // self.batch_size

    def _process_data(self, data_path, hypotheses=None):
        # first try to load vocab
        vocab_file = os.path.join(data_path, "..", "prog_vocab.pkl")
        if os.path.exists(vocab_file):
            vocab = pickle.load(open(vocab_file, 'rb'))
        else:
            # process one
            prog_tokens_file = os.path.join(data_path, "..", "prog_tokens.txt")
            tokens = open(prog_tokens_file).read().splitlines()
            vocab = copy.deepcopy(SPECIAL_TOKENS)
            for t in tokens: vocab[t] = len(vocab)
            pickle.dump(vocab, open(vocab_file, 'wb'))
        
        # try to load training data
        if hypotheses is None:
            train_data_file = os.path.join(data_path, "training.pkl")
        else:
            train_data_file = os.path.join(data_path, "training_reinforce.pkl")
        if os.path.exists(train_data_file) :
            logging.info("Loading data from {}.".format(train_data_file))
            data = pickle.load(open(train_data_file, 'rb'))
            return data, vocab
        
        logging.info("Processed data not exists. Processing it now.")
        data = []
        # read generated boards
        board_paths = open(os.path.join(data_path, "train_board_files.txt")).read().splitlines()
        #programs = open(os.path.join(data_path, "train_programs.txt")).read().splitlines()
        programs = json.load(open(os.path.join(data_path, "train_programs.json")))
        for i, (board_file, program) in enumerate(zip(board_paths, programs)):
            board = read_board(open(os.path.join(data_path, board_file)).read())
            prog_tokens = []
            #program = [program]
            for p in program:
                p = p.replace("(", " ( ").replace(")", " ) ").split()
                prog_tokens.append([vocab[w] for w in p] + [SPECIAL_TOKENS['<eos>']])
            if hypotheses is None:
                data.append((board, prog_tokens))
            else:
                valid_ids = hypotheses.observe(np.array(board) - 2)
                data.append((board, prog_tokens, valid_ids))
            print("\r {} / {} finished.".format(i + 1, len(programs)), end='', flush=True)
        print("\r")

        logging.info("Number of training data: {}".format(len(data)))
        pickle.dump(data, open(train_data_file, 'wb'))
        return data, vocab


class BoardLoader(object):
    def __init__(self, data_path, batch_size, hypotheses, drop_last=False):
        self.data, self.vocab = self._process_data(data_path, hypotheses)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def _wrap_data(self, batch_data):
        # wrap data with numpy array
        boards, contexts = [], []
        for b, c in batch_data:
            boards.append(b)
            contexts.append(c)
        
        return torch.LongTensor(boards).unsqueeze(1), contexts

    def __iter__(self):
        """
        return a list of np arrays (boards)
        """
        random.shuffle(self.data)
        batch_data, contexts = [], []
        for b, c in self.data:
            batch_data.append((b, c))
            if len(batch_data) == self.batch_size:
                yield self._wrap_data(batch_data)
                batch_data = []
        if (not self.drop_last) and (len(batch_data) > 0):
            yield self._wrap_data(batch_data)

    def __len__(self):
        if self.drop_last:
            return len(self.data) // self.batch_size
        else:
            return len(self.data  + self.batch_size - 1) // self.batch_size

    def _process_data(self, data_path, hypotheses):
        # first try to load vocab
        vocab_file = os.path.join(data_path, "..", "prog_vocab.pkl")
        if os.path.exists(vocab_file):
            vocab = pickle.load(open(vocab_file, 'rb'))
        else:
            # process one
            prog_tokens_file = os.path.join(data_path, "..", "prog_tokens.txt")
            tokens = open(prog_tokens_file).read().splitlines()
            vocab = copy.deepcopy(SPECIAL_TOKENS)
            for t in tokens: vocab[t] = len(vocab)
            pickle.dump(vocab, open(vocab_file, 'wb'))
        
        # try to load training data
        train_data_file = os.path.join(data_path, "training_reinforce.pkl")
        if os.path.exists(train_data_file) :
            logging.info("Loading data from {}.".format(train_data_file))
            data = pickle.load(open(train_data_file, 'rb'))
            return data, vocab

        logging.info("Processed data not exists. Processing it now.")
        data = []
        # read generated boards
        board_paths = open(os.path.join(data_path, "train_board_files.txt")).read().splitlines()
        for i, board_file in enumerate(board_paths):
            board = read_board(open(os.path.join(data_path, board_file)).read())
            valid_ids = hypotheses.observe(np.array(board) - 2)
            data.append((board, valid_ids))
            print("\r {} / {} finished.".format(i + 1, len(board_paths)), end='', flush=True)
        print("\r")

        logging.info("Number of training data: {}".format(len(data)))
        pickle.dump(data, open(train_data_file, 'wb'))
        return data, vocab