import numpy as np
from tqdm import tqdm
import pickle
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ====== Dataloader with PCA Support ======
class Dataloader:
    def __init__(self, normalization=False, min_samples_per_class=50, n_components=80):
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


# ====== MLP with Plotting ======
class MLP:
    def __init__(self, dataloader, epoch: int, learning_rate: float, gamma=1, hidden_dim=64, batch_size=32):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epoch = epoch
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

        self.dataloader = dataloader
        self.inter_variable = {}
        self.weights1_list = []

    def initialize_weights(self, input_dim):
        W1 = np.random.randn(input_dim, self.hidden_dim) * 0.01
        W2 = np.random.randn(self.hidden_dim + 1, self.dataloader.n_classes) * 0.01
        return W1, W2

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, x, W1, W2, no_gradient=False):
        z1 = W1.T @ x
        a1 = self.relu(z1)
        a1 = np.vstack((a1, np.ones((1, a1.shape[1]))))
        z2 = W2.T @ a1
        z2 = np.clip(z2, -50, 50)
        a2 = 1 / (1 + np.exp(-z2))
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

        delta1 = self.relu_derivative(z1) * (W2[:-1, :] @ delta2)
        grad_W1 = x @ delta1.T

        return grad_W1, grad_W2

    def update_weights(self, W1, W2, grad_W1, grad_W2, learning_rate):
        return W1 - learning_rate * grad_W1, W2 - learning_rate * grad_W2

    def update_learning_rate(self, lr):
        return lr * self.gamma

    @staticmethod
    def accuracy(label, y_hat):
        y_hat = y_hat.T
        return (y_hat.argmax(axis=1) == label.argmax(axis=1)).mean()

    @staticmethod
    def loss(output, label):
        return np.sum((output.T - label) ** 2) / (2 * label.shape[0])

    def train(self):
        n_classes = self.dataloader.n_classes

        for i in range(5):
            lr = self.learning_rate
            X_train, y_train, X_test, y_test = self.dataloader.get_Train_test_data()
            X_train = np.c_[X_train, np.ones(X_train.shape[0])].T
            X_test = np.c_[X_test, np.ones(X_test.shape[0])].T
            y_train = self.dataloader.one_hot_mapping(y_train, n_classes)
            y_test = self.dataloader.one_hot_mapping(y_test, n_classes)

            W1, W2 = self.initialize_weights(X_train.shape[0])

            temp_train_acc, temp_train_loss = [], []
            temp_test_acc, temp_test_loss = [], []

            for e in tqdm(range(self.epoch), desc=f'Fold {i + 1}/5'):
                train_epoch_acc = []
                train_epoch_loss = []

                indices = np.random.permutation(X_train.shape[1])
                for b in range(0, X_train.shape[1], self.batch_size):
                    idx = indices[b:b + self.batch_size]
                    xb = X_train[:, idx]
                    yb = y_train[idx]
                    out = self.forward(xb, W1, W2)
                    gW1, gW2 = self.back_prop(yb, W2)
                    W1, W2 = self.update_weights(W1, W2, gW1, gW2, lr)

                    train_epoch_acc.append(self.accuracy(yb, out))
                    train_epoch_loss.append(self.loss(out, yb))

                pred = self.forward(X_test, W1, W2, no_gradient=True)
                temp_test_acc.append(self.accuracy(y_test, pred))
                temp_test_loss.append(self.loss(pred, y_test))

                lr = self.update_learning_rate(lr)

                temp_train_acc.append(np.mean(train_epoch_acc))
                temp_train_loss.append(np.mean(train_epoch_loss))

                lr = self.update_learning_rate(lr)

            self.weights1_list.append((W1, W2))
            self.train_accuracy.append(temp_train_acc)
            self.train_loss.append(temp_train_loss)
            self.test_accuracy.append(temp_test_acc)
            self.test_loss.append(temp_test_loss)

        print(f"Average test accuracy: {np.mean([acc[-1] for acc in self.test_accuracy]):.4f}")

        self.plot_metrics()

    def plot_metrics(self):
        train_acc = np.array(self.train_accuracy[0])
        test_acc = np.array(self.test_accuracy[0])
        train_loss = np.array(self.train_loss[0])
        test_loss = np.array(self.test_loss[0])

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(test_acc, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_loss, label='Train Loss')
        plt.plot(test_loss, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        plt.tight_layout()
        plt.show()


# ====== Run Training ======
if __name__ == "__main__":
    dataloader = Dataloader(normalization=True, n_components=100)
    model = MLP(dataloader=dataloader, epoch=200, learning_rate=0.001, gamma=0.9995, hidden_dim=128, batch_size=32)
    model.train()
