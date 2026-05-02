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

# 🔥 UPGRADE DATA
X = X[:10000]
y = y[:10000]

X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ======================
# MODEL TRANSFORMER (UPGRADE)
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
# TRAINING
# ======================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 🔥 UPGRADE EPOCH
epochs = 20

batch_size = 64

for epoch in range(epochs):
    model.train()
    
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        output = model(X_batch)
        loss = criterion(output, y_batch)

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

y_true = y_test.numpy()
y_pred = pred.numpy()

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)

print("\n===== TRANSFORMER (UPGRADED) =====")
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
plt.title("Transformer (Improved) vs Actual")
plt.show()