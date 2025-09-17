"""
Scaler Dosyalarını Oluştur
API için gerekli scaler'ları kaydet
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Gerçek veri aralıkları (notebook'tan alındı)
real_features = np.array([
    [2.455679, -2.029098, 23.214802],  # Min değerler
    [4.222920, 0.007496, 41.450232]    # Max değerler
])

real_target = np.array([[0.0], [1.0]])  # SOC 0-1 arası

# Scaler'ları oluştur
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Fit et
scaler_features.fit(real_features)
scaler_target.fit(real_target)

# Kaydet
os.makedirs("src/api", exist_ok=True)
joblib.dump(scaler_features, "src/api/scaler_features.pkl")
joblib.dump(scaler_target, "src/api/scaler_target.pkl")

print("✅ Scaler dosyaları oluşturuldu:")
print("   - src/api/scaler_features.pkl")
print("   - src/api/scaler_target.pkl")

# Test et
loaded_scaler_features = joblib.load("src/api/scaler_features.pkl")
loaded_scaler_target = joblib.load("src/api/scaler_target.pkl")

print("\n✅ Scaler dosyaları test edildi!")
print(f"   Features scaler range: {loaded_scaler_features.data_range_}")
print(f"   Target scaler range: {loaded_scaler_target.data_range_}")
