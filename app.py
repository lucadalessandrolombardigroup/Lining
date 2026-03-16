import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# --------------------------------------------------
# CACHE MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model_and_scalers():

    model = tf.keras.models.load_model("ANN_model/ann_surrogate.keras")
    scaler_X = joblib.load("ANN_model/scaler_X.pkl")
    scaler_y = joblib.load("ANN_model/scaler_y.pkl")

    return model, scaler_X, scaler_y


# --------------------------------------------------
# ANALYTICAL MODEL (for validation)
# --------------------------------------------------

R = 5
psi = 0
p = 0


def compute_d_r_R_permille(Erm, nu, sig_0, phi_rad, eta, psi, p):
    k = (1 + np.sin(phi_rad)) / (1 - np.sin(phi_rad))

    coh = (2 * sig_0 * np.tan(phi_rad)) / (
        (1 + k) * ((eta + 1) ** (k - 1) - 2 / (1 + k))
    )

    # Radial displacement for MC criterion
    Rpl = (2 / (1 + k) * (coh / np.tan(phi_rad) + sig_0) / (coh / np.tan(phi_rad) + p)) ** (1 / (k - 1))

    if Rpl < 1:
        d_r_R = 1e3 * (1 + nu) / Erm * (sig_0 - p)  # [permille]
    else:
        lam = (
            ((1 - nu ** 2) * (k ** 2 - 1) * (coh / np.tan(phi_rad) + p))
            / (Erm * (2 + (k - 1) * (1 - np.sin(psi))))
            * (Rpl ** (2 / (1 - np.sin(psi)) + k - 1) * 1 ** (-2 / (1 - np.sin(psi))) - 1 ** (k - 1))
        )
        eps_tan_p = lam * (1 - np.sin(psi))
        eps_tan_e = -(1 + nu) / Erm * (
            (coh / np.tan(phi_rad) + sig_0) * (1 - 2 * nu)
            - (coh / np.tan(phi_rad) + p) * 1 ** (k - 1) * (k * (1 - nu) - nu)
        )
        d_r_R = 1e3 * (eps_tan_p + eps_tan_e)  # [permille]

    return d_r_R


model, scaler_X, scaler_y = load_model_and_scalers()

# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.markdown(
    "<h1>⛰️ Ground convergence predictor</h1>",
    unsafe_allow_html=True
)

st.write(
"""Enter the following parameters: 
"""
)

# --------------------------------------------------
# INPUT PARAMETERS
# --------------------------------------------------

nu = st.number_input(
    "Poisson ratio ν",
    min_value=0.0,
    value=0.2,
    step = 0.05
)

phi_deg = st.number_input(
    "Friction angle φ (degrees)",
    min_value=0.0,
    value=30.0,
    step = 5.0
)

eta = st.number_input(
    "Normalized plastic zone η",
    min_value=0.0,
    value=2.0,
    step = 0.5
)

# --------------------------------------------------
# CONVERT φ
# --------------------------------------------------

phi_rad = np.radians(phi_deg)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("Predict"):

    X = np.array([[nu, phi_rad, eta]])
    X_scaled = scaler_X.transform(X)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    Erm_ref = 1e6      # kPa
    sig0_ref = 2.5e3   # kPa

    d_r_R_permille = compute_d_r_R_permille(
        Erm_ref, nu, sig0_ref, phi_rad, eta, psi, p
    )
    y_true = d_r_R_permille * Erm_ref / sig0_ref
    
    #st.latex(rf"\frac{{\delta_r}}{{R}} \cdot \frac{{E}}{{\sigma_0}} = {y_pred[0][0]:.2f}")

    error_percent = 100 * abs(y_pred[0][0] - y_true) / abs(y_true)

    st.markdown("### Validation against analytical solution")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("ANN prediction", f"{y_pred[0][0]:.2f}")

    with col2:
        st.metric("Analytical value", f"{y_true:.2f}")

    with col3:
        st.metric("Error (%)", f"{error_percent:.3f}")