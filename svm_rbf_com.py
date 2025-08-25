import os
import pickle
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# === Step 1: Load Data ===
with open('lfw.pkl', 'rb') as file:
    flw_people = pickle.load(file)

X, Y = flw_people['data'], flw_people['target']
images = flw_people['images']
target_names = flw_people['target_names']

# === Step 2: Filter Classes with >= 50 Samples ===
counts = Counter(Y)
valid_ids = [i for i, c in counts.items() if c >= 50]
mask = np.isin(Y, valid_ids)
X = X[mask]
Y = Y[mask]
images = images[mask]

# Relabel classes
label_map = {old: new for new, old in enumerate(sorted(set(Y)))}
Y = np.array([label_map[y] for y in Y])
target_names = [target_names[i] for i in sorted(valid_ids)]

# === Step 3: Normalize and Apply PCA ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=50, whiten=True, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# === Step 4: Split Data ===
x_train, x_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.2, random_state=42)

# === Step 5: Generate Performance with Different Regularization Parameters (C) ===
C_values = [0.1, 1, 10, 50, 100]
train_accuracies = []
test_accuracies = []

for C in C_values:
    # SVM with RBF Kernel
    model_rbf = svm.SVC(kernel='rbf', C=C, gamma=0.01)
    model_rbf.fit(x_train, y_train)
    train_accuracy = model_rbf.score(x_train, y_train)
    test_accuracy = model_rbf.score(x_test, y_test)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# === Step 6: Generate Confusion Matrix for RBF Kernel ===
# 使用最佳参数重新训练模型（例如 C=50.0, gamma=0.01）
best_model_rbf = svm.SVC(kernel='rbf', C=50.0, gamma=0.01)
best_model_rbf.fit(x_train, y_train)
y_pred_best = best_model_rbf.predict(x_test)

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap='viridis', xticks_rotation=45)
plt.title("Confusion Matrix of SVM (RBF Kernel) on LFW Dataset")
plt.tight_layout()
plt.show()


# === Step 6: Plot the Regularization Parameter (C) Comparison ===
plt.figure(figsize=(10, 5))
plt.plot(C_values, train_accuracies, label="Train Accuracy", marker='o')
plt.plot(C_values, test_accuracies, label="Test Accuracy", marker='s')
plt.xscale('log')
plt.xlabel('C (log scale)')
plt.ylabel('Accuracy')
plt.title('SVM Accuracy vs C (RBF Kernel)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 7: Performance with Different Loss Functions (MSE vs CrossEntropy) ===
# For simplicity, we use SVM with linear kernel to simulate the effect of different loss functions
# This part is more theoretical, as SVM does not directly support MSE or CrossEntropy loss.
# You can compare them by using different SVM implementations or by setting up multi-class classification

# Display Example Image
plt.imshow(images[4], cmap='gray')
plt.title(f"Sample Image: {target_names[Y[0]]}")
plt.axis('off')
plt.show()
