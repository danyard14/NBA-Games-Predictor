import numpy as np
import torch


def normalize(v):
    """
    Normalize the vector
    :param v: vector to normalize
    :return normalized vector v
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def get_acc(out, Y):
    """
    Calculate the NN accuracy
    :param out: the output of the NN after forwarded X in each layer at NN
    :param Y: a matrix of size lxm, where Y[i,:] is c_i (indicator vector for label i)
    :return the accuracy NN percentage
    """
    if type(torch.zeros(0)) == type(out):
        out = out.detach().numpy()
        Y = Y.detach().numpy()
    preds = np.argmax(out, axis=0)
    true_labels = np.argmax(Y, axis=0)
    acc = sum(preds == true_labels)
    return acc / Y.shape[1]


def boolean_to_one_hot(Y):
    return np.where(Y == [True], np.array([0, 1]), np.array([1, 0]))


def boolean_to_one_hot_torch(Y):
    return torch.where(Y == [True], torch.tensor([0, 1]), torch.tensor([1, 0]))
