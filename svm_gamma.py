import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# === Step 1: Load and filter LFW data ===
with open('lfw.pkl', 'rb') as file:
    flw_people = pickle.load(file)

X, Y = flw_people['data'], flw_people['target']
target_names = flw_people['target_names']
counts = Counter(Y)
valid_ids = [i for i, c in counts.items() if c >= 50]
mask = np.isin(Y, valid_ids)
X, Y = X[mask], Y[mask]

# Remap class indices to 0 ~ n-1
label_map = {old: new for new, old in enumerate(sorted(set(Y)))}
Y = np.array([label_map[y] for y in Y])
target_names = [target_names[i] for i in sorted(valid_ids)]

# === Step 2: Normalize and apply PCA ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=50, whiten=True, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# === Step 3: Split data ===
x_train, x_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.2, random_state=42)

# === Step 4: Define hyperparameter grid ===
C_values = [0.01, 0.1, 1, 5, 10, 50, 100]
gamma_values = ['scale', 0.1, 0.01, 0.001]
poly_degrees = [2, 3, 4, 5]

# === Step 5: Evaluate and store results ===
results_poly = {}  # {(gamma, degree): [(C, acc), ...]}
results_rbf = {}   # {gamma: [(C, acc), ...]}

for gamma in gamma_values:
    accs_rbf = []
    for C in C_values:
        clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        accs_rbf.append((C, acc))
        print(f'RBF | gamma={gamma}, C={C} => acc={acc:.4f}')
    results_rbf[gamma] = accs_rbf

    for d in poly_degrees:
        accs_poly = []
        for C in C_values:
            clf = svm.SVC(kernel='poly', degree=d, C=C, gamma=gamma)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            acc = metrics.accuracy_score(y_test, y_pred)
            accs_poly.append((C, acc))
            print(f'Poly | gamma={gamma}, degree={d}, C={C} => acc={acc:.4f}')
        results_poly[(gamma, d)] = accs_poly

# === Step 6: Plot results ===
for gamma in gamma_values:
    plt.figure(figsize=(12, 6))
    # RBF line
    C_rbf, acc_rbf = zip(*results_rbf[gamma])
    plt.plot(C_rbf, acc_rbf, marker='o', label=f'RBF (gamma={gamma})', linewidth=2)

    for d in poly_degrees:
        C_poly, acc_poly = zip(*results_poly[(gamma, d)])
        plt.plot(C_poly, acc_poly, marker='s', linestyle='--', label=f'Poly d={d}, gamma={gamma}')

    plt.xscale('log')
    plt.xlabel('C (log scale)')
    plt.ylabel('Accuracy')
    plt.title(f'SVM Accuracy vs C @ gamma={gamma}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'svm_grid_gamma_{gamma}.png')
    plt.show()
