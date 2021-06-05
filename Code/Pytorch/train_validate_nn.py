import numpy as np
import torch
from torch.nn import ReLU
from torch.nn.functional import cross_entropy
from torch.optim import optimizer

import data_loader
from net import TorchNetwork
from NeuralNetwork.utils import boolean_to_one_hot
from Utils.names import *


def train_pytorch_nn(function_object, num_layers, epochs, batch_size, lr):
    X_train, Y_train = data_loader.encode_data(TRAIN_DATA_PATH, TRAIN_ALLSTAR_PATH, TRAIN_STANDINGS_PATH)
    X_test, Y_test = data_loader.encode_data(TEST_DATA_PATH, TEST_ALLSTAR_PATH, TEST_STANDINGS_PATH)
    X_train, Y_train = np.array(X_train).T, np.array(Y_train).T
    X_test, Y_test = np.array(X_test).T, np.array(Y_test).T
    Y_train, Y_test = Y_train.reshape(Y_train.shape[0], 1), Y_test.reshape(Y_test.shape[0], 1)
    Y_train, Y_test = boolean_to_one_hot(Y_train), boolean_to_one_hot(Y_test)
    Y_train = Y_train.T
    Y_test = Y_test.T
    input_size = X_train.shape[0]
    m = X_train.shape[1]
    num_of_classes = Y_train.shape[0]
    network = TorchNetwork(num_layers, input_size, num_of_classes, function_object)
    optimizer = torch.optim.SGD(network.parameters(), lr)
    for epoch in range(epochs):
        perm_indices = np.random.permutation(m)
        for j in range(0, m, batch_size):
            X_batch = X_train[:, perm_indices[j:j + batch_size]]
            Y_batch = Y_train[:, perm_indices[j:j + batch_size]]
            preds = network(X_batch)
            loss = cross_entropy(preds, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_accuracy[epoch] += get_acc(probabilities, Y_batch)

        losses[epoch] /= (m // batch_size)
        training_accuracy[epoch] /= (m // batch_size)


if __name__ == '__main__':
    num_layers = [1, 3, 5, 7, 10]
    epochs = [50, 100, 150]
    batch_sizes = [1, 10, 15, 20]
    lrs = [0.01, 0.001, 0.0001]
    best_accuracy = 0
    best_params = None
    function_objects = [ReLU, torch.tanh]
    for function_object in function_objects:
        for num_layer in num_layers:
            for epoch in epochs:
                for batch_size in batch_sizes:
                    for lr in lrs:
                        print(str(function_object), num_layer, epoch, batch_size, lr)

                        print("Accuracy=", accuracy)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = (num_layer, epoch, batch_size, lr)
    print("Best accuracy: ", best_accuracy, " Achieved with: ", best_params)
    plt.show()

# for batch in train_loader: # Get Batch
#     images, labels = batch
#
#     preds = network(images) # Pass Batch
#     loss = F.cross_entropy(preds, labels) # Calculate Loss
#
#     optimizer.zero_grad()
#     loss.backward() # Calculate Gradients
#     optimizer.step() # Update Weights
#
#     total_loss += loss.item()
#     total_correct += get_num_correct(preds, labels)
#
# print(
#     "epoch:", 0,
#     "total_correct:", total_correct,
#     "loss:", total_loss
# )
