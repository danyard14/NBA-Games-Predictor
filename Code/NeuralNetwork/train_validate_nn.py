
import matplotlib.pyplot as plt
from network import *
from optimizers import *
from utils import get_acc, boolean_to_one_hot
from Code import data_loader
from Utils.names import *

_, axis1 = plt.subplots(1, 1)
_, axis2 = plt.subplots(1, 1)
_, axis3 = plt.subplots(1, 1)


def train_network(num_layers=1, batch_size: int = 32, lr: float = 0.001, epochs: int = 100,
                  policy="increase", activation= ReLU, layers_arr = []):
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
    net = NeuralNetwork(input_size, num_layers, num_of_classes, policy=policy, activation=activation)
    optimizer = SGD(net, lr)
    losses = np.zeros(epochs)
    validation_accuracy = np.zeros(epochs)
    training_accuracy = np.zeros(epochs)
    for epoch in range(epochs):
        net.train_mode()
        perm_indices = np.random.permutation(m)
        for j in range(0, m, batch_size):
            X_batch = X_train[:, perm_indices[j:j + batch_size]]
            Y_batch = Y_train[:, perm_indices[j:j + batch_size]]
            out = net.forward_pass(X_batch)
            loss, probabilities = net.soft_max_layer.soft_max(out, Y_batch)
            net.backward_pass()
            optimizer.step()
            losses[epoch] += loss
            training_accuracy[epoch] += get_acc(probabilities, Y_batch)

        losses[epoch] /= (m // batch_size)
        training_accuracy[epoch] /= (m // batch_size)
        validation_accuracy[epoch] = validate(net, X_test, Y_test)

        # print(f"epochs = {epoch}, loss = {losses[epoch]}, validation_accuracy = {validation_accuracy[epoch]}"
        #       f" train_accuracy = {training_accuracy[epoch]}")

    axis1.plot(np.arange(0, epochs, 1), training_accuracy)
    axis1.set_xlabel("epochs")
    axis1.set_ylabel("score")
    axis1.legend(layers_arr)
    axis1.set_title(f"training accuracy : batchsize = {batch_size} lr = {lr}")

    axis2.plot(np.arange(0, epochs, 1), validation_accuracy)
    axis2.set_xlabel("epochs")
    axis2.set_ylabel("score")
    axis2.legend(layers_arr)
    axis2.set_title(f"validation accuracy : batchsize = {batch_size} lr = {lr}")

    axis3.plot(np.arange(0, epochs, 1), losses)
    axis3.set_xlabel("epochs")
    axis3.set_ylabel("score")
    axis3.legend(layers_arr)
    axis3.set_title(f"loss: batchsize = {batch_size} lr = {lr}")
    return max(validation_accuracy)


def validate(net: NeuralNetwork, X_test, Y_test):
    net.eval_mode()
    out = net.forward_pass(X_test)
    _, probabilities = net.soft_max_layer.soft_max(out, Y_test)
    acc = get_acc(probabilities, Y_test)
    return acc


if __name__ == '__main__':
    num_layers = [1, 3, 5, 7, 10]
    epochs = [50, 100, 150]
    batch_sizes = [1, 10, 15, 20]
    lrs = [0.01, 0.001, 0.0001]
    best_accuracy = 0
    best_params = None
    functions = [tanh]
    for function in functions:
        for num_layer in num_layers:
            for epoch in epochs:
                for batch_size in batch_sizes:
                    for lr in lrs:
                        print(str(function), num_layer, epoch, batch_size, lr)
                        accuracy = train_network(num_layers=num_layer, batch_size=batch_size, epochs=epoch, lr=lr,activation=function)
                        print("Accuracy=", accuracy)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = (num_layer, epoch, batch_size, lr)
    print("Best accuracy: ", best_accuracy, " Achieved with: ", best_params)
    plt.show()
