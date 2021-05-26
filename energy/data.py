import os
import eig
from eig.battleship import BattleshipHypothesisSpace, Parser, Executor
from eig.battleship.program import DataType
import numpy as np
import torch
try:
    from .complexity import Complexity
except:
    from complexity import Complexity

BOARD_TO_ID = {'H': -1, 'W': 0, 'B': 1, 'R': 2, 'P': 3}
def read_board(board_raw):
    board = []
    for row in board_raw.splitlines():
        board.append([BOARD_TO_ID[x] for x in row.split(',')])
    return np.array(board)

def append_eig_features(prog, features, eig_context):
    prog_ast = Parser.parse(prog)
    executor = Executor(prog_ast)
    score = eig.compute_eig(executor, eig_context)
    features.append(score)
    features.append(int(score < 1e-6))

def append_nonboard_features(prog, features, complexity):
    append_complex_feats(prog, features, complexity)
    prog_ast = Parser.parse(prog)
    append_ans_type_feats(prog_ast, features)
    append_relev_feats(prog_ast, features)
    

def append_complex_feats(prog, features, complexity):
    features.append(complexity.compute(prog))

def append_ans_type_feats(prog, features):
    if prog.dtype == DataType.BOOLEAN or prog.dtype == DataType.ORIENTATION:
        features.extend([1, 0, 0, 0])
    elif prog.dtype == DataType.NUMBER:
        features.extend([0, 1, 0, 0])
    elif prog.dtype == DataType.COLOR:
        features.extend([0, 0, 1, 0])
    elif prog.dtype == DataType.LOCATION:
        features.extend([0, 0, 0, 1])

def append_relev_feats(prog, features):
    def traverse(node):
        if node.ntype.endswith("_fn"): return 1
        if node.children:
            for c in node.children:
                ret = traverse(c)
                if ret == 1: return 1
        return 0
    features.append(traverse(prog))


def prepare_bag():
    programs = open("./energy_bag_progs.txt").read().splitlines()
    complexity = Complexity("grammar.txt")
    hypotheses = BattleshipHypothesisSpace(grid_size=6, ship_labels=[1, 2, 3], 
                ship_sizes=[2, 3, 4], orientations=['V', 'H'])

    # compute EIG features
    for i in range(18, 0, -1):
        print("Processing board {}".format(i))
        board = read_board(open("../../data/3-export_data/board_{}.txt".format(i)).read())
        belief = eig.Bayes(hypotheses)
        context = eig.Context(hypotheses, belief)
        context.observe(board)
        eig_feats = []
        for j, prog in enumerate(programs):
            prog_feats = []
            question = Parser.parse(prog)
            executor = Executor(question)
            score = eig.compute_eig(executor, context)
            prog_feats.append(score)
            prog_feats.append(int(score < 1e-6))
            eig_feats.append(prog_feats)
            if j % 100 == 0:
                print("\r{} / {}".format(j + 1, len(programs)), end='', flush=True)
        print("\r", end='', flush=True)
        eig_feats = torch.Tensor(eig_feats)
        torch.save(eig_feats, "bagfeats/eig_board_{}.tensor".format(i))

    # compute board-irrelevent features
    nonboard_feats = []
    for i, prog in enumerate(programs):
        prog_feats = []
        append_nonboard_features(prog, prog_feats, complexity)
        nonboard_feats.append(prog_feats)
        if i % 100 == 0:
            print("\r{} / {}".format(i + 1, len(programs)), end='', flush=True)
    nonboard_feats = torch.Tensor(nonboard_feats)
    torch.save(nonboard_feats, "bagfeats/non_board.tensor")
    print("\rBoard irrelevent features generated", flush=True)


def prepare_train():
    complexity = Complexity("grammar.txt")
    hypotheses = BattleshipHypothesisSpace(grid_size=6, ship_labels=[1, 2, 3], 
                ship_sizes=[2, 3, 4], orientations=['V', 'H'])
    train_feats = []
    for i in range(18):
        print(i + 1, end=' ', flush=True)
        board = read_board(open("train/board_{}.txt".format(i + 1)).read())
        programs = open("train/questions_programs_{}.txt".format(i + 1)).read().splitlines()
        belief = eig.Bayes(hypotheses)
        context = eig.Context(hypotheses, belief)
        context.observe(board)
        for i, prog in enumerate(programs):
            prog_feats = []
            append_eig_features(prog, prog_feats, context)
            append_nonboard_features(prog, prog_feats, complexity)
            train_feats.append(prog_feats)
    train_feats = torch.Tensor(train_feats)
    torch.save(train_feats, "train/all_features.tensor")
    print("\rTraining features generated", flush=True)


def train_data():
    # load precalculated features
    nonboard_feats = torch.load("bagfeats/non_board.tensor")
    bag_feats = []
    for i in range(18):
        eig_feats = torch.load("bagfeats/eig_board_{}.tensor".format(i + 1))
        eig_feats[:, 1] = (eig_feats[:, 0] < 1e-6).to(torch.float)
        bag_feats.append(torch.cat([eig_feats, nonboard_feats], dim=1))
    bag_loglikelihood = - nonboard_feats[:, 0:1].repeat([18, 1])
    bag_feats = torch.cat(bag_feats, dim=0)
    print("Precalculated features loaded, size ", bag_feats.size())

    # load train features
    train_feats = torch.load("train/all_features.tensor")
    print("Training features loaded, size ", train_feats.size())

    # feature rescale
    minv, _ = bag_feats.min(dim=0)
    maxv, _ = bag_feats.max(dim=0)
    bag_feats = (bag_feats - minv) / (maxv - minv)
    train_feats = (train_feats - minv) / (maxv - minv)

    return train_feats, bag_feats, bag_loglikelihood


def prepare_infer_nonboard(progs_ref):
    nonboard_feats = []
    curdir = os.path.dirname(os.path.abspath(__file__))
    complexity = Complexity(os.path.join(curdir, "grammar.txt"))
    for i, prog in enumerate(progs_ref):
        prog_feats = []
        append_nonboard_features(prog, prog_feats, complexity)
        nonboard_feats.append(prog_feats)
    nonboard_feats = torch.Tensor(nonboard_feats)

    bag_nb_feats = torch.load(os.path.join(curdir, "bagfeats/non_board.tensor"))
    bag_feats = []
    for i in range(18):
        eig_feats = torch.load(os.path.join(curdir, "bagfeats/eig_board_{}.tensor".format(i + 1)))
        eig_feats[:, 1] = (eig_feats[:, 0] < 1e-6).to(torch.float)
        bag_feats.append(torch.cat([eig_feats, bag_nb_feats], dim=1))
    bag_feats = torch.cat(bag_feats, dim=0)
    minv, _ = bag_feats.min(dim=0)
    maxv, _ = bag_feats.max(dim=0)

    return nonboard_feats, minv, maxv

if __name__ == "__main__":
    prepare_train()
