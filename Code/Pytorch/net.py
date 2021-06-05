import torch.nn as nn
import numpy as np


class TorchNetwork(nn.Module):
    def __init__(self, num_layers, input_size, out_size=2, activation_function=nn.ReLU):
        super(TorchNetwork, self).__init__()
        layers = [nn.Linear(input_size, 6), activation_function()] + list(
            np.ndarray([[nn.Linear(2 * (i + 2), 2 * (i + 3)), activation_function()] for i in
                        range(1, num_layers // 2)]).flatten()) \
                 + list(np.ndarray([[nn.Linear(2 * (i + 3), 2 * (i + 2)), activation_function()] for i in
                                    range(num_layers // 2 - 1, 0, -1)]).flatten())
        layers += nn.Linear(layers[-1].out_features, out_size)
        layers += nn.ReLU
        self.linear_stack = nn.Sequential(*layers)
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(input_size, input_size * 2),
        #     activation_function(),
        #     nn.Linear(input_size * 2, input_size),
        #     activation_function(),
        #     nn.Linear(input_size, out_size),
        #     nn.ReLU()
        # )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits
