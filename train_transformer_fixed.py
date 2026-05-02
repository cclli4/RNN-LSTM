import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from preprocess import load_ecg, normalize, create_dataset

# ======================
# LOAD DATA
# ======================
path = "mit-bih-arrhythmia-database-1.0.0/100"

ecg = load_ecg(path)
ecg = normalize(ecg)

X, y = create_dataset(ecg, seq_len=100)

# ======================
# 🔥 FIX: SPLIT TIME SERIES (JAUH)
# ======================

# train di awal
X_train = X[:8000]
y_train = y[:8000]

# test jauh dari train (biar ga leakage)
X_test = X[20000:30000]
y_test = y[20000:30000]

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# convert ke tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ======================
# MODEL TRANSFORMER
# ======================
class TimeSeriesTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_projection = nn.Linear(1, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        out = self.fc_out(x[:, -1, :])
        return out

model = TimeSeriesTransformer()

# ======================
# TRAINING SETUP
# ======================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
batch_size = 64

train_losses = []

# ======================
# TRAINING LOOP
# ======================
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        output = model(X_batch)
        loss = criterion(output, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= (len(X_train) / batch_size)
    train_losses.append(epoch_loss)

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.6f}")

# ======================
# EVALUATION
# ======================
model.eval()
with torch.no_grad():
    pred = model(X_test)

y_true = y_test.numpy()
y_pred = pred.numpy()

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)

print("\n===== TRANSFORMER (FIXED) =====")
print("MSE :", mse)
print("RMSE:", rmse)
print("MAE :", mae)

# ======================
# PLOT PREDICTION
# ======================
plt.figure(figsize=(12,5))
plt.plot(y_true[:200], label="Actual")
plt.plot(y_pred[:200], label="Prediction")
plt.legend()
plt.title("Transformer (Fixed) vs Actual")
plt.show()

# ======================
# PLOT LOSS
# ======================
plt.figure()
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()