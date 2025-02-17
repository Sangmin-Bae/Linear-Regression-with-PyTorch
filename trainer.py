import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as func

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        super().__init__()

    def train(self, x, y, config):
        lowest_loss = np.inf
        best_model = None

        for epoch_idx in range(config.n_epochs):
            y_hat = self.model(x)
            loss = func.mse_loss(y_hat, y)

            # Initialize gradients
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Descent
            self.optimizer.step()

            if loss < lowest_loss:
                lowest_loss = loss
                best_model = deepcopy(self.model.state_dict())

            if (epoch_idx + 1) % config.interval == 0:
                print("Epoch %d: loss=%.4e" % (epoch_idx + 1, float(loss)))

        # Save Best Model
        self.model.load_state_dict(best_model)