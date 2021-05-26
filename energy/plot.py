import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json
import numpy as np
import sys

thetas, loglik = json.load(open(sys.argv[1]))
thetas = np.array(thetas)[:, 0, :]
names = ["eig", "eig_01", "comp", "type_bool", "type_num", "type_color", "type_loc", "relev"]

def imsave(path, values, ylabel):
    fig, axes = plt.subplots(2, 4)
    for i, n in enumerate(names):
        ax = axes[i % 2][i // 2]
        ax.plot(values[:, i], label=n)
        ax.legend()
    fig.suptitle(ylabel)
    fig.savefig(path)
    

#imsave("thetas_{}.png".format(sys.argv[1]), thetas, "thetas")
#print(thetas[-1])
t = (thetas[-1] * -100).tolist()
for t, n in zip(t, names):
    print("{:10s} {:.3f}".format(n, t))

fig, ax = plt.subplots()
ax.plot(loglik, label='log-likelihood')
ax.legend()
fig.savefig("loglikelihood_{}.png".format(sys.argv[1]))
#imsave("fvalues_{}.png".format(sys.argv[1]), fvalues, "fvalues")
