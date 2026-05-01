import wfdb
import numpy as np
import matplotlib.pyplot as plt

# path ke file (tanpa ekstensi)
record_path = "mit-bih-arrhythmia-database-1.0.0/100"

# baca sinyal ECG
record = wfdb.rdrecord(record_path)

# ambil signal (biasanya ada 2 channel)
signal = record.p_signal

print("Shape signal:", signal.shape)

# ambil 1 channel saja (univariate)
ecg = signal[:, 0]

print("Contoh data:", ecg[:10])

# plot biar ngerti bentuknya
plt.plot(ecg[:1000])
plt.title("ECG Signal (First 1000 samples)")
plt.show()