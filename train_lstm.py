import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from preprocess import load_ecg, normalize, create_dataset, train_test_split

# ======================
# LOAD DATA
# ======================
path = "mit-bih-arrhythmia-database-1.0.0/100"

ecg = load_ecg(path)
ecg = normalize(ecg)

X, y = create_dataset(ecg, seq_len=100)

# ambil subset biar cepat (MWE)
X = X[:20000]
y = y[:20000]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# convert ke tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ======================
# MODEL LSTM
# ======================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel()

# ======================
# TRAINING SETUP
# ======================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ======================
# TRAINING LOOP
# ======================
epochs = 20

for epoch in range(epochs):
    model.train()

    output = model(X_train)
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# ======================
# EVALUATION
# ======================
model.eval()
with torch.no_grad():
    pred = model(X_test)

# ubah ke numpy
y_true = y_test.numpy()
y_pred = pred.numpy()

# ======================
# METRIK
# ======================
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)

print("\n===== EVALUATION =====")
print("MSE :", mse)
print("RMSE:", rmse)
print("MAE :", mae)

# ======================
# PLOT
# ======================
plt.figure(figsize=(12,5))

plt.plot(y_true[:200], label="Actual")
plt.plot(y_pred[:200], label="Prediction")

plt.legend()
plt.title("LSTM Prediction vs Actual (ECG)")
plt.xlabel("Time Step")
plt.ylabel("Normalized Value")

plt.show()