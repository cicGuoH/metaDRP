# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSLRGradientDescentLearningRule(nn.Module):
    """Simple (stochastic) gradient descent learning rule.
    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
        p[i] := p[i] - learning_rate * dE/dp[i]
    With `learning_rate` a positive scaling parameter.
    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, device, num_inner_updates, use_learnable_learning_rates, init_learning_rate=1e-3):
        """Creates a new learning rule object.
        Args:
            init_learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
        """
        super(LSLRGradientDescentLearningRule, self).__init__()
        print(init_learning_rate)
        assert init_learning_rate > 0., 'learning_rate should be positive.'

        self.init_learning_rate = torch.ones(1) * init_learning_rate
        self.init_learning_rate.to(device)
        self.num_inner_updates = num_inner_updates
        self.use_learnable_learning_rates = use_learnable_learning_rates

    def initialise(self, fast_params):
        self.names_learning_rates = nn.ParameterList()
        for idx, param in enumerate(fast_params):
            self.names_learning_rates.append(nn.Parameter(data=torch.ones(self.num_inner_updates + 1) * self.init_learning_rate,
                requires_grad=self.use_learnable_learning_rates))

    def update_params(self, fast_params, names_grads_wrt_params, num_step, tau=0.1):
        """Applies a single gradient descent update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        new_fast_params = []
        for (idx, (param, grad)) in enumerate(zip(fast_params, names_grads_wrt_params)):
            if param.requires_grad:
                if grad is None:
                    grad = torch.tensor(1., device = param.device)
                new_fast_params.append(param-self.names_learning_rates[idx][num_step]*grad)
            else:
                new_fast_params.append(param)
        return new_fast_params