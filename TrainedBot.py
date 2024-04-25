import random

import numpy as np
import pandas as pd
import torch

from BotModel import BotModel
import torch.optim as optim
from torch import nn
import torch.utils.data as data_utils
from matplotlib import pyplot as plt
from Ship import get_ship


def load_process_training_data(train_data_path: str):
    df = pd.read_csv(train_data_path)
    df = df.dropna()
    train_x = df.drop('Optimal_Direction', axis=1).drop('Unnamed: 0', axis=1)
    train_y = df['Optimal_Direction']
    train_y = pd.get_dummies(train_y)
    train_x = torch.from_numpy(train_x.values).float()
    train_x = train_x / 10
    train_y = torch.from_numpy(train_y.values).float()
    return train_x, train_y


def train(data_path):
    model = BotModel()
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    loss_func = nn.CrossEntropyLoss()
    train_x, train_y = load_process_training_data(data_path)
    epochs = 10000
    print(f'Shape of train x is {train_x.shape}')
    print(f'Shape of train y is {train_y.shape}')
    losses = []
    for i in range(epochs):
        optimizer.zero_grad()
        logits = model(train_x)
        loss = loss_func(logits, train_y)
        probs = get_probs(logits)
        print(f'Epoch Number:{i}')
        print(f'Calculated loss between logits and train y:{loss.item()}')
        losses.append(loss.item())
        acc = torch.sum(torch.argmax(train_y, dim=1) == torch.argmax(probs, dim=1)) / train_y.shape[0]
        print(f'Accuracy:{acc}')
        loss.backward()
        optimizer.step()
    plot_loss_by_epochs(losses)


def plot_loss_by_epochs(losses):
    losses = losses[2:]
    epochs = [i for i in range(len(losses))]
    plt.plot(epochs, losses)
    plt.show()


def get_probs(logits):
    return nn.Softmax(dim=1)(logits)


# def process_data():

if __name__ == '__main__':
    train('train_data.csv')
