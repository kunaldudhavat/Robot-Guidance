import numpy as np
import torch
from torch import nn


class BotModel(nn.Module):
    def __init__(self):
        super(BotModel, self).__init__()
        self.input_layer = nn.Linear(in_features=4, out_features=20, bias=True)
        self.hidden_layer = nn.Linear(in_features=20, out_features=25, bias=True)
        self.hidden_layer1 = nn.Linear(in_features=25, out_features=30, bias=True)
        self.hidden_layer2 = nn.Linear(in_features=30, out_features=25, bias=True)
        self.hidden_layer3 = nn.Linear(in_features=25, out_features=20, bias=True)
        self.output_layer = nn.Linear(in_features=20, out_features=9, bias=True)
        self.activation_function = nn.Sigmoid()

    def forward(self, input_tensor):
        output = self.input_layer(input_tensor)
        output = self.activation_function(output)
        output = self.hidden_layer(output)
        output = self.activation_function(output)
        output = self.hidden_layer1(output)
        output = self.activation_function(output)
        output = self.hidden_layer2(output)
        output = self.activation_function(output)
        output = self.hidden_layer3(output)
        output = self.activation_function(output)
        output = self.output_layer(output)
        # probs = nn.Softmax(dim=1)(input)
        return output
