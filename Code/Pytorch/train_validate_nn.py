import numpy as np
import torch
from torch.nn import ReLU
from torch.nn.functional import cross_entropy

import data_loader
from net import TorchNetwork
from NeuralNetwork.utils import get_acc
from Utils.names import *


def boolean_to_one_hot_torch(Y):
    one_hot = torch.zeros(Y.shape[0], 2)
    for i, true_false in enumerate(Y):
        if true_false[0].item() is True:
            one_hot[i] = torch.tensor([0, 1])
        else:
            one_hot[i] = torch.tensor([1, 0])
    return one_hot


def train_pytorch_nn(function_object, num_layers, epochs, batch_size, lr):
    X_train, Y_train = data_loader.encode_data(TRAIN_DATA_PATH, TRAIN_ALLSTAR_PATH, TRAIN_STANDINGS_PATH)
    X_test, Y_test = data_loader.encode_data(TEST_DATA_PATH, TEST_ALLSTAR_PATH, TEST_STANDINGS_PATH)
    X_train, Y_train = torch.tensor(X_train).T, torch.tensor(Y_train).T
    X_test, Y_test = torch.tensor(X_test).T, torch.tensor(Y_test).T
    Y_train, Y_test = Y_train.reshape(Y_train.shape[0], 1), Y_test.reshape(Y_test.shape[0], 1)
    Y_train, Y_test = boolean_to_one_hot_torch(Y_train), boolean_to_one_hot_torch(Y_test)
    Y_train = Y_train.T
    Y_test = Y_test.T
    input_size = X_train.shape[0]
    m = X_train.shape[1]
    num_of_classes = Y_train.shape[0]
    network = TorchNetwork(num_layers, input_size, num_of_classes, function_object)
    optimizer = torch.optim.SGD(network.parameters(), lr)
    losses = np.zeros(epochs)
    validation_accuracy = np.zeros(epochs)
    training_accuracy = np.zeros(epochs)
    for epoch in range(epochs):
        perm_indices = np.random.permutation(m)
        for j in range(0, m, batch_size):
            X_batch = X_train[:, perm_indices[j:j + batch_size]]
            Y_batch = torch.transpose(Y_train[:, perm_indices[j:j + batch_size]], 0, 1)
            X_batch = torch.transpose(X_batch, 0, 1).float()
            preds = network(X_batch)
            # preds = preds.T
            loss = cross_entropy(preds, torch.tensor(Y_batch.argmax(1)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_accuracy[epoch] += get_acc(torch.transpose(preds, 0, 1), torch.transpose(Y_batch, 0, 1))
        # UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
        #   loss = cross_entropy(preds, torch.tensor(Y_batch.argmax(1)))
        losses[epoch] /= (m // batch_size)
        training_accuracy[epoch] /= (m // batch_size)
        validation_accuracy[epoch] = validate_torch_nn(network, X_test, Y_test)

    return max(validation_accuracy)


def validate_torch_nn(network: TorchNetwork, X_test, Y_test):
    network.eval()
    out = network(torch.transpose(X_test.float(), 0, 1))
    acc = get_acc(torch.transpose(out, 0, 1), Y_test)
    network.train()
    return acc


if __name__ == '__main__':
    print(torch.cuda.is_available())
    f = open("outputs.txt", "a")
    num_layers = [1, 3, 5, 7]
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
                        f.write(str(function_object) + "," + str(num_layer) + "," + str(epoch) + "," + str(batch_size) + "," + str(lr) + "\n")
                        accuracy = train_pytorch_nn(function_object, num_layer, epoch, batch_size, lr)
                        print("Accuracy=", accuracy)
                        f.write("Accuracy="+str(accuracy) + "\n")
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = (num_layer, epoch, batch_size, lr)
    print("Best accuracy: ", best_accuracy, " Achieved with: ", best_params)
    f.close()
    # plt.show()

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
