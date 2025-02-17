import torch
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        super().__init__()

        # Single Fully Connected Layer
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # |x| = (N, input_size)
        y = self.linear(x)
        # |y| = (N, output_size)

        return y