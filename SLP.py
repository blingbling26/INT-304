import random
import time
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Set random seed
random.seed(int(time.time()))
np.random.seed(int(time.time()))

# === DataLoader class using lfw.pkl ===
class Dataloader:
    def __init__(self, normalization=False, min_samples_per_class=50):
        with open('lfw.pkl', 'rb') as f:
            raw_data = pickle.load(f)
        X = raw_data['data']
        y = raw_data['target']
        target_names = raw_data['target_names']

        # Filter by minimum sample count
        counts = np.bincount(y)
        valid_classes = np.where(counts >= min_samples_per_class)[0]
        mask = np.isin(y, valid_classes)
        X, y = X[mask], y[mask]

        # Remap labels
        label_map = {old: new for new, old in enumerate(valid_classes)}
        y = np.array([label_map[yy] for yy in y])

        if normalization:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        self.data = X
        self.labels = y
        self.target_names = [target_names[i] for i in valid_classes]
        self.n_classes = len(valid_classes)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.folds = list(skf.split(X, y))
        self.index = 0

    def get_Train_test_data(self):
        train_idx, test_idx = self.folds[self.index]
        self.index = (self.index + 1) % len(self.folds)
        return self.data[train_idx], self.labels[train_idx], self.data[test_idx], self.labels[test_idx]

    def one_hot_mapping(self, labels: np.ndarray, num_classes: int):
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels] = 0.99
        return one_hot

# === SLP implementation ===
class SLP:
    def __init__(self, dataloader: Dataloader, epoch: int, learning_rate: float, gamma=1):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epoch = epoch

        self.weights1_list = []
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

        self.dataloader = dataloader
        self.inter_variable = {}

    def initialize_weights(self, input_dim, output_dim):
        return np.random.rand(input_dim, output_dim) * 2 - 1

    def forward(self, x, w1, no_gradient=False):
        z1 = w1.T.dot(x)
        a1 = 1 / (1 + np.exp(-z1))
        if no_gradient:
            return a1
        self.inter_variable = {"z1": z1, "a1": a1}
        return a1

    def back_prop(self, x, y):
        delta_k = self.inter_variable["a1"] - y.T
        delta_j = self.inter_variable["a1"] * (1 - self.inter_variable["a1"]) * delta_k
        gradient1 = x.dot(delta_j.T)
        return gradient1

    def update_weight(self, w1, gradient1, learning_rate):
        return w1 - learning_rate * gradient1, learning_rate

    def update_learning_rate(self, learning_rate):
        return learning_rate * self.gamma

    @staticmethod
    def accuracy(label, y_hat: np.ndarray):
        y_hat = y_hat.T
        acc = y_hat.argmax(axis=1) == label.argmax(axis=1)
        return acc.mean()

    @staticmethod
    def loss(output, label):
        return np.sum(((output.T - label) ** 2)) / (2 * label.shape[0])

    def train(self):
        start = time.time()
        for i in range(5):
            with tqdm(total=self.epoch) as _tqdm:
                _tqdm.set_description(f'Fold {i+1}/5')

                learning_rate = self.learning_rate
                X_train, y_train, X_test, y_test = self.dataloader.get_Train_test_data()

                # Add bias term
                X_train = np.c_[X_train, np.ones(X_train.shape[0])].T
                X_test = np.c_[X_test, np.ones(X_test.shape[0])].T

                y_train = self.dataloader.one_hot_mapping(y_train, self.dataloader.n_classes)
                y_test = self.dataloader.one_hot_mapping(y_test, self.dataloader.n_classes)

                w1 = self.initialize_weights(X_train.shape[0], self.dataloader.n_classes)

                temp_test_accuracy = []
                for e in range(self.epoch):
                    for k in range(X_train.shape[1]):
                        x = X_train[:, k].reshape((-1, 1))
                        y = y_train[k].reshape(1, -1)
                        out = self.forward(x, w1, no_gradient=False)
                        gW1 = self.back_prop(x, y)
                        w1, learning_rate = self.update_weight(w1, gW1, learning_rate)

                    pred = self.forward(X_test, w1, no_gradient=True)
                    acc = self.accuracy(y_test, pred)
                    loss = self.loss(pred, y_test)
                    temp_test_accuracy.append(acc)

                    learning_rate = self.update_learning_rate(learning_rate)
                    _tqdm.set_postfix(acc=f'{acc:.4f}', loss=f'{loss:.4f}')
                    _tqdm.update(1)

                self.weights1_list.append(w1)
                self.test_accuracy.append(temp_test_accuracy)

        avg_acc = np.mean([acc[-1] for acc in self.test_accuracy])
        print(f"Average test accuracy: {avg_acc:.4f}")
        print(f"Trained time: {1000 * (time.time() - start):.2f} ms")

# Run
if __name__ == '__main__':
    dataloader = Dataloader(normalization=True)
    model = SLP(dataloader=dataloader, epoch=200, learning_rate=0.01, gamma=0.999)
    model.train()
