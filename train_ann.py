import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# =========================================================
# 1) LOAD DATA
# =========================================================

df = pd.read_csv("dataset.csv")

# Target normalizzato
df["y_norm"] = df["d_r_R_permille"] * df["Erm_kPa"] / df["sig_0_kPa"]

# Inputs
X = df[["nu", "phi_rad", "eta"]].values.astype(np.float32)   #vedere se mettere 64
y = df["y_norm"].values.astype(np.float32)

# =========================================================
# 2) SCALE INPUTS
# =========================================================

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.reshape(-1,1)).flatten()

# =========================================================
# 3) BUILD MODEL
# =========================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(1, activation="linear")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

# =========================================================
# 4) TRAIN
# =========================================================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="loss",   
    patience=150,           
    restore_best_weights=True
)

history = model.fit(
    X_scaled,
    y_scaled,
    epochs=2000,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# =========================================================
# 5) PLOT TRAINING LOSS
# =========================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training MSE")

if "val_loss" in history.history:
    plt.plot(history.history["val_loss"], label="Validation MSE")

plt.xlabel("Epoch")
plt.ylabel("MSE (scaled target)")
plt.title("Training Loss")
plt.yscale("log")  # scala log molto utile
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()


# =========================================================
# 6) SAVE MODEL AND SCALERS
# =========================================================

# Salva modello
model.save("ann_surrogate.keras")

# Salva scaler input
import joblib
joblib.dump(scaler_X, "scaler_X.pkl")

# Salva scaler target
joblib.dump(y_scaler, "scaler_y.pkl")

print("Modello e scaler salvati.")

