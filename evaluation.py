import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# ================= LOAD DATA =================
df = pd.read_csv("credit_default_preprocessed.csv")
target_col = 'default' if 'default' in df.columns else df.columns[-1]

X = df.drop(columns=[target_col]).values
y = df[target_col].values

# Convert labels to {-1, +1}
y = np.where(y == 0, -1, 1)

# ================= PREPROCESS =================
X = SimpleImputer(strategy='mean').fit_transform(X)

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= SCALE =================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================= INITIALIZE =================
w = np.zeros(X_train.shape[1])
b = 0

lr = 0.01
lambda_reg = 0.01
epochs = 20

loss_history = []
acc_history = []

# ================= TRAIN (SGD SVM) =================
for epoch in range(epochs):
    loss = 0

    for i in range(len(X_train)):
        condition = y_train[i] * (np.dot(w, X_train[i]) + b) >= 1

        if condition:
            w -= lr * (2 * lambda_reg * w)
        else:
            w -= lr * (2 * lambda_reg * w - y_train[i] * X_train[i])
            b -= lr * (-y_train[i])
            loss += max(0, 1 - y_train[i] * (np.dot(w, X_train[i]) + b))

    # Accuracy tracking
    y_temp = np.sign(np.dot(X_test, w) + b)
    acc = np.mean(y_temp == y_test)

    loss_history.append(loss)
    acc_history.append(acc)

# ================= PREDICTION =================
y_pred = np.sign(np.dot(X_test, w) + b)

# Convert to 0/1
y_test_bin = np.where(y_test == -1, 0, 1)
y_pred_bin = np.where(y_pred == -1, 0, 1)

# For ROC/PR → use decision scores
scores = np.dot(X_test, w) + b

# ================= METRICS =================
print("\n===== METRICS =====")
print("Accuracy:", accuracy_score(y_test_bin, y_pred_bin))
print("Precision:", precision_score(y_test_bin, y_pred_bin))
print("Recall:", recall_score(y_test_bin, y_pred_bin))
print("F1 Score:", f1_score(y_test_bin, y_pred_bin))
print("ROC-AUC:", roc_auc_score(y_test_bin, scores))

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(y_test_bin, y_pred_bin)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# ================= ROC CURVE =================
fpr, tpr, _ = roc_curve(y_test_bin, scores)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1])
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

# ================= PRECISION-RECALL CURVE =================
p, r, _ = precision_recall_curve(y_test_bin, scores)

plt.figure()
plt.plot(r, p)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# ================= ACCURACY vs EPOCHS =================
plt.figure()
plt.plot(range(epochs), acc_history)
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# ================= LOSS CURVE =================
plt.figure()
plt.plot(range(epochs), loss_history)
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()