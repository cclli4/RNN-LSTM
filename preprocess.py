# import wfdb
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# def load_ecg(record_path):
#     record = wfdb.rdrecord(record_path)
#     signal = record.p_signal
#     ecg = signal[:, 0]  # ambil 1 channel
#     # ecg = signal
#     return ecg

# def normalize(data):
#     scaler = MinMaxScaler()
#     data = data.reshape(-1, 1)
#     return scaler.fit_transform(data)

# # def normalize(data):
# #     scaler = MinMaxScaler()
# #     return scaler.fit_transform(data)  # langsung (N, 2)

# def create_dataset(data, seq_len=100):
#     X, y = [], []
#     for i in range(len(data) - seq_len):
#         X.append(data[i:i+seq_len])
#         y.append(data[i+seq_len])
#     return np.array(X), np.array(y)

# # def create_dataset(data, seq_len=100):
# #     X, y = [], []
# #     for i in range(len(data) - seq_len):
# #         X.append(data[i:i+seq_len])
# #         y.append(data[i+seq_len])  # sekarang 2 nilai
# #     return np.array(X), np.array(y)

# if __name__ == "__main__":
#     path = "mit-bih-arrhythmia-database-1.0.0/100"

#     ecg = load_ecg(path)
#     ecg = normalize(ecg)

#     X, y = create_dataset(ecg, seq_len=100)

#     print("X shape:", X.shape)
#     print("y shape:", y.shape)

# def train_test_split(X, y, train_ratio=0.8):
#     split = int(len(X) * train_ratio)
#     return X[:split], X[split:], y[:split], y[split:]
    
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# print("Train:", X_train.shape)
# print("Test:", X_test.shape)


import wfdb
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_ecg(record_path):
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal
    ecg = signal[:, 0]
    return ecg

def normalize(data):
    scaler = MinMaxScaler()
    data = data.reshape(-1, 1)
    return scaler.fit_transform(data)

def create_dataset(data, seq_len=100):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

def train_test_split(X, y, train_ratio=0.8):
    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]

if __name__ == "__main__":
    path = "mit-bih-arrhythmia-database-1.0.0/100"

    ecg = load_ecg(path)
    ecg = normalize(ecg)

    X, y = create_dataset(ecg)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print("X:", X.shape)
    print("Train:", X_train.shape)