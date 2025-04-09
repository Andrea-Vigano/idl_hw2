from config import default_mlp_config
from mlp import MLP
from optimizers import GD, MomentumGD, Adam, Optimizer
from tensor import Tensor
from losses import binary_cross_entropy, l2
from numpyNN import sample_data, plot_loss, plot_decision_boundary
import matplotlib.pyplot as plt
import numpy as np


def get_accuracy(y_true, y_predicted):
    assert y_true.shape == y_predicted.shape

    correct = 0
    for (y_sample, y_pred_sample) in zip(y_true, y_predicted):
        if y_sample == y_pred_sample:
            correct += 1
    return correct / len(y_true)


def train(model, dataset: str, n_epochs: int, optimizer: Optimizer, loss_function, logs):
    print_frequency = 100
    X_train, y_train, X_test, y_test = sample_data(dataset)

    # Deliverable 7 - add non-linear feature for XOR
    # Will add x * y
    # product_feature_train = (X_train[:, 0] * X_train[:, 1]).reshape(-1, 1)
    # product_feature_test = (X_test[:, 0] * X_test[:, 1]).reshape(-1, 1)
    # X_train = np.hstack((X_train, product_feature_train))
    # X_test = np.hstack((X_test, product_feature_test))

    # Deliverable 7 - add non-linear feature for swiss-roll
    # sin_feature_train = np.sin(X_train[:, 1]).reshape(-1, 1)
    # sin_feature_test = np.sin(X_test[:, 1]).reshape(-1, 1)
    # X_train = np.hstack((X_train, sin_feature_train))
    # X_test = np.hstack((X_test, sin_feature_test))


    X_train, y_train, X_test, y_test = (Tensor(X_train, requires_gradient=False),
                                        Tensor(y_train, requires_gradient=False),
                                        Tensor(X_test, requires_gradient=False),
                                        Tensor(y_test, requires_gradient=False))

    for epoch in range(n_epochs):
        # Train
        y_predicted = model(X_train)

        optimizer.zero_grad()
        loss = loss_function(y_predicted, y_train)

        loss.backward()
        optimizer.step()

        logs['train_loss'].append(loss.data[0])

        # Validate
        y_predicted = model(X_test)
        loss = loss_function(y_predicted, y_test)
        logs['test_loss'].append(loss.data[0])

        if (epoch + 1) % print_frequency == 0:
            print(f"Epoch {epoch + 1} -> train_loss={logs['train_loss'][-1]}; val_loss={logs['test_loss'][-1]}")


if __name__ == '__main__':
    logs = {
        'train_loss': [],
        'test_loss': [],
    }

    dataset = 'swiss-roll'
    mlp = MLP(config=default_mlp_config)

    n_epochs = 40000
    optimizer = Adam(mlp.parameters, learning_rate=0.05)

    # Deliverable 2 training configs
    # n_epochs = 10000
    # optimizer = GD(mlp.parameters, learning_rate=0.02)

    # Deliverable 3 training configs
    # n_epochs = 10000
    # optimizer = Adam(mlp.parameters, learning_rate=0.02)

    # Deliverable 4 training configs (regressor and classifier)
    # n_epochs = 10000
    # optimizer = Adam(mlp.parameters, learning_rate=0.02)

    # Deliverable 5 training configs
    # learning_rate=0.02
    # learning_rate=0.001 for tuned momentum

    # Deliverable 6 training configs
    # n_epochs = 10000
    # optimizer = Adam(mlp.parameters, learning_rate=0.01)

    # Deliverable 7 training configs - XOR
    # n_epochs = 10000
    # optimizer = Adam(mlp.parameters, learning_rate=0.005)

    # Deliverable 7 training configs - swiss-roll
    # n_epochs = 40000
    # optimizer = Adam(mlp.parameters, learning_rate=0.05)

    # Train the model
    train(mlp, dataset, n_epochs, optimizer, binary_cross_entropy, logs)

    X_train, y_train, X_test, y_test = sample_data(dataset)

    # Deliverable 7 - add non-linear feature for XOR
    # Will add x * y
    # product_feature_train = (X_train[:, 0] * X_train[:, 1]).reshape(-1, 1)
    # product_feature_test = (X_test[:, 0] * X_test[:, 1]).reshape(-1, 1)
    # X_train = np.hstack((X_train, product_feature_train))
    # X_test = np.hstack((X_test, product_feature_test))

    # Deliverable 7 - add non-linear feature for swiss-roll
    # sin_feature_train = np.sin(X_train[:, 1]).reshape(-1, 1)
    # sin_feature_test = np.sin(X_test[:, 1]).reshape(-1, 1)
    # X_train = np.hstack((X_train, sin_feature_train))
    # X_test = np.hstack((X_test, sin_feature_test))

    plot_decision_boundary(X_test, y_test, lambda _X: (mlp(Tensor(_X)).data > 0.5).astype(int))
    plt.show()
    plot_loss(logs)
    plt.show()
