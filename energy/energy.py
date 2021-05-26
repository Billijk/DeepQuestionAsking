import os
import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn, optim
import torch.nn.functional as F
from random import sample
import json


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Linear(8, 1, bias=False)
    
    def forward(self, features):
        if not isinstance(features, torch.Tensor):
            features = torch.Tensor(features)
        return self.weights(features)

    def backward(self, feats, N, sampled_bag, loglikelihood):
        bag_energy = self.weights(sampled_bag).detach()
        log_w = -bag_energy - loglikelihood                              # TODO: do we need a negative sign here and line 27?
        w_normalized = torch.exp(log_w - float(logsumexp(log_w)))
        bag_feats = torch.sum(sampled_bag * w_normalized, keepdim=True, dim=0)
        self.weights.weight.grad = -N * bag_feats + feats
        return w_normalized.sum()


def train(save_dir, iters=100000, save_iter=10000, lr=0.01):
    from data import train_data
    train_feats, bag_feats, bag_loglikelihood = train_data()
    train_feats = train_feats.cuda()
    bag_feats = bag_feats.cuda()
    bag_loglikelihood = bag_loglikelihood.cuda()

    model = EnergyModel()
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.95 ** (iter // 1000))
    logger_theta, logger_ll = [], []

    print("Start training!")
    last_theta = model.weights.weight.data.cpu().clone()
    loglikelihood = 0
    for i in range(iters):
        scheduler.step()
        features = train_feats

        # forward pass
        energy = model(features)

        # backward pass
        optimizer.zero_grad()
        Z = model.backward(features.sum(dim=0), features.size(0), bag_feats, bag_loglikelihood)
        optimizer.step()
        #print("[Iter {}]".format(i + 1), model.weights.weight.grad)

        loglikelihood += (-energy - torch.log(Z)).sum().cpu().item()
        #print((-energy - torch.log(Z)).sum())

        if (i + 1) % 100 == 0:
            #print("[Iter {}]".format(i + 1), model.weights.weight.data.cpu() - last_theta)
            print("[Iter {}] Loglikelihood {:.3f}".format(i + 1, loglikelihood / 100))
            last_theta = model.weights.weight.data.cpu().clone()
            logger_theta.append(last_theta.numpy().tolist())
            logger_ll.append(loglikelihood / 100)
            loglikelihood = 0
        if (i + 1) % save_iter == 0:
            torch.save(model.state_dict(), save_dir + '/it_{}.pth'.format(i + 1))
    
    json.dump([logger_theta, logger_ll], open("train_neg_sgd_{}_decay".format(lr), "w"))


def infer(features, model=None):
    if model is None:
        model = EnergyModel()
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        "weights.pth"), map_location=lambda storage, loc: storage))
    return model(features).detach().view(-1).numpy()


if __name__ == "__main__":
    #import sys
    #train("save", 100000, lr=float(sys.argv[1]))
    model = EnergyModel()
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                    "weights.pth"), map_location=lambda storage, loc: storage))
    print(model.weights.weight.data)


"""
class_imp <- R6Class(
  "imp",
  public = list(
    bag = NULL,
    energies = NULL,
    logw = NULL,
    logw_unnormalized = NULL,
    thetas = NULL,
    history = NULL,
    N_emp = NULL,
    emp.f = NULL,
    emp.f.rel = NULL,
    sim.f = NULL,
    learning_rate = NULL,
    gradient = NULL,
    delta = NULL,
    i = 0,
    adagrad = FALSE,
    init = function() {
      self$thetas <- runif(n = length(self$emp.f), min = -.5, max = .5)
      self$history <- NULL
    },
    update_logw = function() {
      log_w_u <- compute_log_weight_unnormalized(self$energies, self$bag$loglik, self$bag$n)
      ## compute_log_weight_unnormalized is actually energies - loglik
      log_w <- log_w_u - matrixStats::logSumExp(log_w_u) # normalizing in log space
      self$logw_unnormalized <- log_w_u
      self$logw <- log_w
    },
    update_sim.f = function() {
      if (nrow(self$bag$f.matrix) != length(self$logw)) browser()
      # old.sim.f <- self$sim.f
      w <- exp(self$logw)
      # new.sim.f <- apply(self$bag$f.matrix, 2, FUN = weighted.mean, w) # slow
      # new.sim.f <- colWeightedMeans(self$bag$f.matrix, w) # fast
      new.sim.f <- colWeightedMeans_fast(self$bag$f.matrix, w) # fastest
      #       if (isTRUE(all.equal(old.sim.f, new.sim.f))) {
      #         self$i <- self$i + 1
      #         if (self$i >= 40) stop("Learning limit reached")
      #       }
      self$sim.f <- new.sim.f
    },
    update_gradient = function() {
      self$gradient <-  self$N_emp * self$sim.f - self$emp.f
    },
    update_delta = function() {
      if (self$adagrad) {
        self$delta <- adagrad(self$learning_rate, self$gradient, self$history$gradients)
      } else {
        self$delta <- self$learning_rate * self$gradient
      }
    },
    update_thetas = function() {
      self$thetas <- self$thetas - self$delta
    },
    update_energies = function() {
      # computing the energy of each unique particle
      thetas <- self$thetas
      features <- self$bag$f.matrix
      if (length(thetas) != ncol(features)) browser() # restarting the whole thing has solved it in the past
      self$energies <- energy.loglik(thetas, features)
    },
    one_step = function() {
      # update energies
      self$energies
      self$update_energies()
      self$energies

      # update weights based on energy
      self$logw
      self$update_logw()
      self$logw

      # update sim.f
      self$sim.f
      self$update_sim.f()
      self$sim.f

      # update gradient
      self$gradient
      self$update_gradient()
      self$gradient

      # update delta
      self$delta
      self$update_delta()
      self$delta

      # update thetas
      self$thetas
      self$update_thetas()
      self$thetas

      invisible()
    }
  )
)
"""
