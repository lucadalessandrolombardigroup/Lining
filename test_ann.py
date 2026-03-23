import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

# =========================================================
# 1) LOAD MODEL AND SCALERS
# =========================================================

model = tf.keras.models.load_model("ANN_model/ann_surrogate1.keras")
scaler_X = joblib.load("ANN_model/scaler_X1.pkl")
scaler_y = joblib.load("ANN_model/scaler_y1.pkl")

# =========================================================
# 2) LOAD DATASET
# =========================================================

df = pd.read_csv("dataset.csv")
df["y_norm"] = df["d_r_R_new"] * df["Erm_kPa"] / df["sig_0_kPa"]

# =========================================================
# 3) PREDICTION FUNCTION
# =========================================================

def ann_predict(nu, phi_rad, eta_array):

    Xg = np.zeros((len(eta_array), 3))
    Xg[:, 0] = nu
    Xg[:, 1] = phi_rad
    Xg[:, 2] = eta_array

    Xg_scaled = scaler_X.transform(Xg)
    y_pred_scaled = model.predict(Xg_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    return y_pred.flatten()

# =========================================================
# 4) PLOT ANN vs DATA (curve continue)
# =========================================================

sig_0_list = sorted(df["sig_0_kPa"].unique())
nu_list    = sorted(df["nu"].unique())
Erm_list   = sorted(df["Erm_kPa"].unique())
phi_train   = sorted(df["phi_rad"].unique())
phi_extra_deg = [15, 25, 35, 45, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
#phi_extra_deg = []
phi_extra = np.radians(phi_extra_deg).tolist()
phi_list = phi_train + phi_extra
#phi_list = sorted(df["phi_rad"].unique())
eta_list   = sorted(df["eta"].unique())

# intervallo continuo per curva ANN
eta_interp = np.linspace(min(eta_list), max(eta_list), 200)
#eta_extra  = np.linspace(max(eta_list), max(eta_list), 100)
eta_extra  = np.linspace(max(eta_list), 20.0, 100)

for sig_0 in sig_0_list:

    fig, axes = plt.subplots(3, 3,
                             sharex=True, sharey=True,
                             figsize=(12, 10))

    for i, nu in enumerate(nu_list):
        for j, Erm in enumerate(Erm_list):

            ax = axes[i, j]

            # subset punti reali
            subset = df[(df["sig_0_kPa"] == sig_0) &
                        (df["nu"] == nu) &
                        (df["Erm_kPa"] == Erm)]

            for phi_rad in phi_list:

                # -----------------------
                # 1) PUNTI REALI
                # -----------------------
                data_phi = subset[np.isclose(subset["phi_rad"], phi_rad)]

                ax.scatter(
                    data_phi["eta"],
                    data_phi["y_norm"]
                )

                # -----------------------
                # 2) CURVA CONTINUA ANN
                # -----------------------

                Xg_interp = np.column_stack([
                    np.full_like(eta_interp, nu),
                    np.full_like(eta_interp, phi_rad),
                    eta_interp
                ])

                Xg_scaled = scaler_X.transform(Xg_interp)
                y_pred_scaled = model.predict(Xg_scaled, verbose=0)
                y_interp = scaler_y.inverse_transform(y_pred_scaled)
                
                if phi_rad in phi_train : 
                    line, = ax.plot(
                        eta_interp,
                        y_interp.flatten(),
                        label=f"{np.degrees(phi_rad):.0f}°"
                    )
                
                else : 
                    line, = ax.plot(
                        eta_interp,
                        y_interp.flatten(),
                        linestyle="--",
                        label=f"{np.degrees(phi_rad):.0f}°"
                    )

                Xg_extra = np.column_stack([
                    np.full_like(eta_extra, nu),
                    np.full_like(eta_extra, phi_rad),
                    eta_extra
                ])

                Xg_scaled = scaler_X.transform(Xg_extra)
                y_pred_scaled = model.predict(Xg_scaled, verbose=0)
                y_extra = scaler_y.inverse_transform(y_pred_scaled).flatten()

                ax.plot(
                    eta_extra,
                    y_extra,
                    linestyle="--",
                    color=line.get_color()
                )

            if i == 0:
                ax.set_title(f"Erm = {Erm:.2e} kPa")

            if j == 0:
                ax.set_ylabel(f"nu = {nu}")

            ax.grid(True, linestyle="--", linewidth=0.5)

    fig.suptitle(f'ANN: d_r/R vs eta   (sig_0 = {sig_0:.0f} kPa)', fontsize=14)
    fig.supxlabel("eta [-]")
    fig.supylabel("d_r/R * Erm/sig_0")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               title="phi",
               loc="lower center",
               ncol=len(phi_list),
               bbox_to_anchor=(0.5, 0.04))

    plt.tight_layout(rect=[0.05, 0.08, 0.95, 0.93])
    plt.savefig(f"ANN_sig_0_{sig_0:.0f}_kPa_1.png")
    plt.close()