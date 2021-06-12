import torch.nn as nn
import numpy as np


class TorchNetwork(nn.Module):
    def __init__(self, num_layers, input_size, out_size=2, activation_function=nn.ReLU):
        super(TorchNetwork, self).__init__()
        layers = []
        if num_layers == 1:
            layers = [nn.Linear(input_size, input_size), activation_function()]
        elif num_layers == 3:
            layers = [nn.Linear(input_size, input_size * 2), activation_function(),
                      nn.Linear(input_size * 2, out_size), activation_function()]
        elif num_layers == 5:
            layers = [nn.Linear(input_size, input_size * 2), activation_function(),
                      nn.Linear(input_size * 2, input_size * 4), activation_function(),
                      nn.Linear(input_size * 4, input_size * 2), activation_function(),
                      nn.Linear(input_size * 2, out_size), activation_function()]
        elif num_layers == 7:
            layers = [nn.Linear(input_size, input_size * 2), activation_function(),
                      nn.Linear(input_size * 2, input_size * 4), activation_function(),
                      nn.Linear(input_size * 4, input_size * 8), activation_function(),
                      nn.Linear(input_size * 8, input_size * 4), activation_function(),
                      nn.Linear(input_size * 4, input_size * 2), activation_function(),
                      nn.Linear(input_size * 2, out_size), activation_function()]

        # part2 = [[nn.Linear(2 * (i + 2), 2 * (i + 3)), activation_function()] for i in range(1, num_layers // 2)]
        # if len(part2) > 0:
        #     part2 = list(np.ndarray(part2).flatten())
        #     out_features = part2[-2].out_features
        #
        # part3 = [[nn.Linear(2 * (i + 3), 2 * (i + 2)), activation_function()] for i in
        #          range(num_layers // 2 - 1, 0, -1)]
        # if len(part3) > 0:
        #     part3 = list(np.ndarray(part3).flatten())
        #     out_features = part3[-2].out_features
        #
        # part4 = [nn.Linear(out_features, out_size), activation_function()]
        #
        # layers = part1 + part2 + part3 + part4
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
        logits = self.linear_stack(x)
        return logits
