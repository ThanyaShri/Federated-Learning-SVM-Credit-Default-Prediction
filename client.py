import flwr as fl
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv("credit_default_preprocessed.csv")
target_col = 'default' if 'default' in df.columns else df.columns[-1]

X = df.drop(columns=[target_col]).values
y = df[target_col].values

# Convert labels to {-1, +1}
y = np.where(y == 0, -1, 1)

# Partition data (no overlap)
client_id = int(input("Enter client ID (0-4): "))
splits = np.array_split(np.arange(len(X)), 5)

idx = splits[client_id]
X_client = X[idx]
y_client = y[idx]

# Preprocessing
X_client = SimpleImputer(strategy='mean').fit_transform(X_client)
scaler = StandardScaler()
X_client = scaler.fit_transform(X_client)

# Initialize weights
dim = X_client.shape[1]
w = np.zeros(dim)
b = 0.0

# Hyperparameters
lr = 0.01
lambda_reg = 0.01
epochs = 3

def train_svm(w, b):
    for _ in range(epochs):
        for i in range(len(X_client)):
            x_i = X_client[i]
            y_i = y_client[i]

            condition = y_i * (np.dot(w, x_i) + b) >= 1

            if condition:
                w = w - lr * (2 * lambda_reg * w)
            else:
                w = w - lr * (2 * lambda_reg * w - y_i * x_i)
                b = b - lr * (-y_i)

    return w, b

class SVMClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [w, np.array([b])]

    def set_parameters(self, parameters):
        global w, b
        w = parameters[0]
        b = parameters[1][0]

    def fit(self, parameters, config):
        if parameters:
            self.set_parameters(parameters)

        new_w, new_b = train_svm(w, b)

        return [new_w, np.array([new_b])], len(X_client), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        preds = np.sign(np.dot(X_client, w) + b)
        acc = np.mean(preds == y_client)

        return float(1 - acc), len(X_client), {"accuracy": acc}

if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=SVMClient()
    )