import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# === Step 1: Load Data ===
with open('lfw.pkl', 'rb') as file:
    flw_people = pickle.load(file)

X, Y = flw_people['data'], flw_people['target']
target_names = flw_people['target_names']

# === Step 2: Filter Classes with >= 50 Samples ===
counts = Counter(Y)
valid_ids = [i for i, c in counts.items() if c >= 50]
mask = np.isin(Y, valid_ids)
X = X[mask]
Y = Y[mask]

# Relabel classes
label_map = {old: new for new, old in enumerate(sorted(set(Y)))}
Y = np.array([label_map[y] for y in Y])

# === Step 3: Normalize and Apply PCA ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=50, whiten=True, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# === Step 4: Split Data ===
x_train, x_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.2, random_state=42)

# === Step 5: Test Different Regularization (C) Values ===
C_values = [0.01, 0.1, 1, 10, 50, 100]
train_loss_rbf, test_loss_rbf = [], []
train_loss_poly, test_loss_poly = [], []

for C in C_values:
    # RBF Kernel
    rbf = svm.SVC(kernel='rbf', C=C, gamma=0.01)
    rbf.fit(x_train, y_train)
    train_loss_rbf.append(1 - rbf.score(x_train, y_train))
    test_loss_rbf.append(1 - rbf.score(x_test, y_test))

    # Polynomial Kernel
    poly = svm.SVC(kernel='poly', C=C, gamma=0.01, degree=2)
    poly.fit(x_train, y_train)
    train_loss_poly.append(1 - poly.score(x_train, y_train))
    test_loss_poly.append(1 - poly.score(x_test, y_test))

# === Step 6: Plot ===
plt.figure(figsize=(10, 6))
plt.plot(C_values, train_loss_rbf, label="Train Loss (RBF)", marker='o')
plt.plot(C_values, test_loss_rbf, label="Test Loss (RBF)", linestyle='--', marker='o')
plt.plot(C_values, train_loss_poly, label="Train Loss (Poly d=2)", marker='s')
plt.plot(C_values, test_loss_poly, label="Test Loss (Poly d=2)", linestyle='--', marker='s')
plt.xscale('log')
plt.xlabel("Regularization Parameter C (log scale)")
plt.ylabel("Loss = 1 - Accuracy")
plt.title("SVM Loss Curve Comparison (RBF vs Polynomial)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("svm_loss_comparison.png")
plt.show()
