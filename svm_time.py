import os
import pickle
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time  # ✅ 加入时间测量模块

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

print(f'sizes: x_train: {x_train.shape}, x_test: {x_test.shape}, num_classes: {len(target_names)}')

# === Step 5: SVM with Polynomial Kernel ===
start_poly = time.time()  # ✅ 开始时间
model_poly = svm.SVC(kernel='poly', C=10.0, gamma=0.01, degree=2)
model_poly.fit(x_train, y_train)
y_pred_poly = model_poly.predict(x_test)
end_poly = time.time()  # ✅ 结束时间
poly_time = end_poly - start_poly

print('\n[Polynomial Kernel]')
print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred_poly):.4f}')
print(metrics.classification_report(y_test, y_pred_poly, target_names=target_names))
print(f'Training + Inference Time: {poly_time:.2f} seconds')

# === Step 6: SVM with RBF Kernel ===
start_rbf = time.time()  # ✅ 开始时间
model_rbf = svm.SVC(kernel='rbf', C=50.0, gamma=0.01)
model_rbf.fit(x_train, y_train)
y_pred_rbf = model_rbf.predict(x_test)
end_rbf = time.time()  # ✅ 结束时间
rbf_time = end_rbf - start_rbf

print('\n[RBF Kernel]')
print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred_rbf):.4f}')
print(metrics.classification_report(y_test, y_pred_rbf, target_names=target_names))
print(f'Training + Inference Time: {rbf_time:.2f} seconds')

# === Step 7: Show Example Image ===
plt.imshow(images[4], cmap='gray')
plt.title(f"Sample Image: {target_names[Y[0]]}")
plt.axis('off')
plt.show()

