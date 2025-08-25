import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Dataloader:
    def __init__(self, normalization=True, min_samples_per_class=50, n_components=80):
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

        if normalization:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        self.pca = PCA(n_components=n_components, whiten=True, random_state=42)
        X = self.pca.fit_transform(X)

        self.data = X
        self.labels = y
        self.n_classes = len(set(y))

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.folds = list(skf.split(self.data, self.labels))
        self.index = 0

    def get_Train_test_data(self):
        train_idx, test_idx = self.folds[self.index]
        self.index = (self.index + 1) % len(self.folds)
        return self.data[train_idx], self.labels[train_idx], self.data[test_idx], self.labels[test_idx]

    def one_hot_mapping(self, labels, num_classes):
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    z = np.clip(z, -50, 50)
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def mse_loss(y_hat, y):
    return np.mean(np.sum((y_hat.T - y) ** 2, axis=1)) / 2


def cross_entropy_loss(y_hat, y):
    eps = 1e-12
    return -np.mean(np.sum(y * np.log(y_hat.T + eps), axis=1))


def train_mlp(hidden_dim, batch_size, learning_rate, gamma, epoch, weight_std, loss_type='crossentropy'):
    dataloader = Dataloader()
    n_classes = dataloader.n_classes

    X_train, y_train, X_test, y_test = dataloader.get_Train_test_data()
    X_train = np.c_[X_train, np.ones(X_train.shape[0])].T
    X_test = np.c_[X_test, np.ones(X_test.shape[0])].T
    y_train = dataloader.one_hot_mapping(y_train, n_classes)
    y_test = dataloader.one_hot_mapping(y_test, n_classes)

    input_dim = X_train.shape[0]
    W1 = np.random.randn(input_dim, hidden_dim) * weight_std
    W2 = np.random.randn(hidden_dim + 1, n_classes) * weight_std

    acc_list = []
    loss_list = []

    for e in tqdm(range(epoch), desc=f"Training {loss_type}"):
        indices = np.random.permutation(X_train.shape[1])
        for b in range(0, X_train.shape[1], batch_size):
            idx = indices[b:b + batch_size]
            xb = X_train[:, idx]
            yb = y_train[idx]

            z1 = W1.T @ xb
            a1 = relu(z1)
            a1 = np.vstack((a1, np.ones((1, a1.shape[1]))))
            z2 = W2.T @ a1
            a2 = softmax(z2)

            if loss_type == 'mse':
                delta2 = a2 - yb.T
            elif loss_type == 'crossentropy':
                delta2 = a2 - yb.T
            else:
                raise ValueError("Invalid loss type")

            grad_W2 = a1 @ delta2.T
            delta1 = relu_derivative(z1) * (W2[:-1, :] @ delta2)
            grad_W1 = xb @ delta1.T

            W1 -= learning_rate * grad_W1
            W2 -= learning_rate * grad_W2

        z1_test = W1.T @ X_test
        a1_test = relu(z1_test)
        a1_test = np.vstack((a1_test, np.ones((1, a1_test.shape[1]))))
        z2_test = W2.T @ a1_test
        y_hat = softmax(z2_test)

        acc = np.mean(y_hat.T.argmax(axis=1) == y_test.argmax(axis=1))
        if loss_type == 'mse':
            loss = mse_loss(y_hat, y_test)
        else:
            loss = cross_entropy_loss(y_hat, y_test)
        acc_list.append(acc)
        loss_list.append(loss)
        learning_rate *= gamma

    return acc_list, loss_list


configs = [
    ('lr=0.01', {'learning_rate': 0.01}),
    ('lr=0.001', {'learning_rate': 0.001}),
    ('epoch=100', {'epoch': 100}),
    ('epoch=300', {'epoch': 300}),
    ('batch=16', {'batch_size': 16}),
    ('batch=64', {'batch_size': 64}),
    ('hidden=64', {'hidden_dim': 64}),
    ('hidden=256', {'hidden_dim': 256}),
    ('init=0.01', {'weight_std': 0.01}),
    ('init=0.1', {'weight_std': 0.1})
]

plt.figure(figsize=(12, 6))
for name, config in configs:
    args = {
        'hidden_dim': 128,
        'batch_size': 32,
        'learning_rate': 0.001,
        'gamma': 0.9995,
        'epoch': 200,
        'weight_std': 0.01
    }
    args.update(config)
    acc, loss = train_mlp(**args)
    plt.plot(acc, label=name)

plt.title("Test Accuracy under Different Hyperparameter Settings")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("hyperparam_comparison.png")
plt.show()
