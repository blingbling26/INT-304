import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm

class Dataloader:
    def __init__(self, normalization=False, min_samples_per_class=50, n_components=100):
        with open('lfw.pkl', 'rb') as f:
            raw_data = pickle.load(f)

        X = raw_data['data']
        y = raw_data['target']
        target_names = raw_data['target_names']

        counts = Counter(y)
        valid_classes = [label for label, count in counts.items() if count >= min_samples_per_class]
        mask = np.isin(y, valid_classes)
        X = X[mask]
        y = y[mask]

        label_map = {old: new for new, old in enumerate(sorted(set(y)))}
        y = np.array([label_map[yy] for yy in y])

        self.target_names = [target_names[i] for i in sorted(valid_classes)]
        self.n_classes = len(set(y))

        if normalization:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        self.pca = PCA(n_components=n_components, whiten=True, random_state=42)
        X = self.pca.fit_transform(X)

        self.data = X
        self.labels = y

        self.fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.folds = list(self.fold.split(self.data, self.labels))
        self.index = 0

    def get_Train_test_data(self):
        train_idx, test_idx = self.folds[self.index]
        self.index = (self.index + 1) % len(self.folds)
        return self.data[train_idx], self.labels[train_idx], self.data[test_idx], self.labels[test_idx]

    def one_hot_mapping(self, labels, num_classes):
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot


class MLP:
    def __init__(self, dataloader, epoch=200, learning_rate=0.001, gamma=0.9995, hidden_dim=128, batch_size=32, activation='relu'):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epoch = epoch
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.activation_name = activation

        self.train_accuracy = []
        self.test_accuracy = []

        self.dataloader = dataloader

    def initialize_weights(self, input_dim):
        W1 = np.random.randn(input_dim, self.hidden_dim) * 0.01
        W2 = np.random.randn(self.hidden_dim + 1, self.dataloader.n_classes) * 0.01
        return W1, W2

    def activate(self, z):
        if self.activation_name == 'relu':
            return np.maximum(0, z)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-z))

    def activate_derivative(self, z):
        if self.activation_name == 'relu':
            return (z > 0).astype(float)
        elif self.activation_name == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)

    def softmax(self, z):
        z = np.clip(z, -50, 50)
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward(self, x, W1, W2, no_gradient=False):
        z1 = W1.T @ x
        a1 = self.activate(z1)
        a1 = np.vstack((a1, np.ones((1, a1.shape[1]))))
        z2 = W2.T @ a1
        a2 = self.softmax(z2)

        if no_gradient:
            return a2
        else:
            self.inter_variable = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
            return a2

    def back_prop(self, y, W2):
        a2 = self.inter_variable["a2"]
        a1 = self.inter_variable["a1"]
        x = self.inter_variable["x"]
        z1 = self.inter_variable["z1"]

        delta2 = a2 - y.T
        grad_W2 = a1 @ delta2.T
        delta1 = self.activate_derivative(z1) * (W2[:-1, :] @ delta2)
        grad_W1 = x @ delta1.T

        return grad_W1, grad_W2

    def update_weights(self, W1, W2, grad_W1, grad_W2, lr):
        return W1 - lr * grad_W1, W2 - lr * grad_W2

    def accuracy(self, label, y_hat):
        y_hat = y_hat.T
        return (y_hat.argmax(axis=1) == label.argmax(axis=1)).mean()

    def train(self):
        acc_per_epoch = []

        for fold in range(5):
            X_train, y_train, X_test, y_test = self.dataloader.get_Train_test_data()
            X_train = np.c_[X_train, np.ones(X_train.shape[0])].T
            X_test = np.c_[X_test, np.ones(X_test.shape[0])].T
            y_train = self.dataloader.one_hot_mapping(y_train, self.dataloader.n_classes)
            y_test = self.dataloader.one_hot_mapping(y_test, self.dataloader.n_classes)

            W1, W2 = self.initialize_weights(X_train.shape[0])
            lr = self.learning_rate

            test_acc_list = []

            for epoch in tqdm(range(self.epoch), desc=f"{self.activation_name.upper()} Fold {fold+1}/5"):
                indices = np.random.permutation(X_train.shape[1])
                for b in range(0, X_train.shape[1], self.batch_size):
                    idx = indices[b:b+self.batch_size]
                    xb = X_train[:, idx]
                    yb = y_train[idx]
                    out = self.forward(xb, W1, W2)
                    gW1, gW2 = self.back_prop(yb, W2)
                    W1, W2 = self.update_weights(W1, W2, gW1, gW2, lr)
                pred = self.forward(X_test, W1, W2, no_gradient=True)
                acc = self.accuracy(y_test, pred)
                test_acc_list.append(acc)
                lr *= self.gamma

            acc_per_epoch.append(test_acc_list)

        avg_test_acc = np.mean(acc_per_epoch, axis=0)
        self.test_accuracy = avg_test_acc
        print(f"[{self.activation_name}] Final accuracy: {avg_test_acc[-1]:.4f}")
        return avg_test_acc

    class FlexibleMLP:
        def __init__(self, input_dim, hidden_dim, output_dim,
                     activation='relu', output='softmax', loss_fn='cross_entropy',
                     lr=0.001, gamma=1, epochs=100, batch_size=32):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.activation_type = activation
            self.output_type = output
            self.loss_type = loss_fn
            self.lr = lr
            self.gamma = gamma
            self.epochs = epochs
            self.batch_size = batch_size

            self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
            self.W2 = np.random.randn(hidden_dim + 1, output_dim) * 0.01

            self.train_acc = []
            self.train_loss = []

        def activation(self, z):
            if self.activation_type == 'relu':
                return np.maximum(0, z)
            elif self.activation_type == 'sigmoid':
                return 1 / (1 + np.exp(-z))

        def activation_derivative(self, z):
            if self.activation_type == 'relu':
                return (z > 0).astype(float)
            elif self.activation_type == 'sigmoid':
                sig = 1 / (1 + np.exp(-z))
                return sig * (1 - sig)

        def output_fn(self, z):
            if self.output_type == 'softmax':
                z -= np.max(z, axis=0, keepdims=True)
                exp_z = np.exp(z)
                return exp_z / np.sum(exp_z, axis=0, keepdims=True)
            elif self.output_type == 'sigmoid':
                return 1 / (1 + np.exp(-z))

        def compute_loss(self, y_pred, y_true):
            if self.loss_type == 'cross_entropy':
                y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
                return -np.sum(y_true * np.log(y_pred.T)) / y_true.shape[0]
            elif self.loss_type == 'mse':
                return np.sum((y_pred.T - y_true) ** 2) / (2 * y_true.shape[0])

        def accuracy(self, y_pred, y_true):
            return np.mean(np.argmax(y_pred.T, axis=1) == np.argmax(y_true, axis=1))

        def train(self, X_train, y_train):
            X_train = np.c_[X_train, np.ones(X_train.shape[0])].T
            y_train = y_train
            for epoch in tqdm(range(self.epochs)):
                indices = np.random.permutation(X_train.shape[1])
                for i in range(0, X_train.shape[1], self.batch_size):
                    idx = indices[i:i + self.batch_size]
                    x = X_train[:, idx]
                    y = y_train[idx]

                    # Forward
                    z1 = self.W1.T @ x
                    a1 = self.activation(z1)
                    a1 = np.vstack([a1, np.ones((1, a1.shape[1]))])
                    z2 = self.W2.T @ a1
                    a2 = self.output_fn(z2)

                    # Loss
                    loss = self.compute_loss(a2, y)
                    acc = self.accuracy(a2, y)

                    # Backward
                    if self.loss_type == 'cross_entropy' and self.output_type == 'softmax':
                        delta2 = a2 - y.T
                    else:
                        delta2 = (a2 - y.T) * (a2 * (1 - a2)) if self.output_type == 'sigmoid' else (a2 - y.T)

                    grad_W2 = a1 @ delta2.T
                    delta1 = self.activation_derivative(z1) * (self.W2[:-1] @ delta2)
                    grad_W1 = x @ delta1.T

                    self.W1 -= self.lr * grad_W1
                    self.W2 -= self.lr * grad_W2

                self.train_loss.append(loss)
                self.train_acc.append(acc)
                self.lr *= self.gamma


# Run comparison
if __name__ == "__main__":
    dataloader = Dataloader(normalization=True, n_components=100)

    relu_model = MLP(dataloader, activation='relu')
    relu_acc = relu_model.train()

    sigmoid_model = MLP(dataloader, activation='sigmoid')
    sigmoid_acc = sigmoid_model.train()

    plt.figure()
    plt.plot(relu_acc, label='ReLU')
    plt.plot(sigmoid_acc, label='Sigmoid')
    plt.title("Test Accuracy Comparison of Activation Functions")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("activation_comparison.png")
    plt.show()
