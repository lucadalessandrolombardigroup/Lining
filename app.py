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

psi = 0
p = 0


def compute_d_r_R_new(Erm, nu, gamma, h, R, phi_rad, coh, psi, p):
    
    G = Erm / (2 * (1 + nu))
    k = (1 + np.sin(phi_rad)) / (1 - np.sin(phi_rad))
    k_psi = (1 + np.sin(psi)) / (1 - np.sin(psi))
    
    sig_0 = gamma*h
    
    lam = 1- p/sig_0
    sig_c = (2 * coh * np.cos(phi_rad)) / (1 - np.sin(phi_rad))
    lam_e = ((k - 1) * sig_0 + sig_c) / ((k + 1) * sig_0)


    den_lam_a = k - nu * (k + 1)
    if den_lam_a != 0:
        lam_a = lam_e * ((1 - nu) * (k + 1)) / den_lam_a
    else:
        lam_a = np.nan

    print(f"Lambda: {lam} | Lambda_e: {lam_e} | Lambda_a: {lam_a}")
    Rp = ((2*lam_e)/((k+1)*lam_e - (k-1)*lam)) ** (1 / (k - 1)) # sarebbe Rp /R
    eta = (Rp - R)/R
        
    if lam <= lam_e:
        u_r_R =  (1 + nu) / Erm * (sig_0 - p) 
    
    elif lam_e < lam <= lam_a:
 
        F1 = -(1 - 2*nu) * (k + 1) / (k - 1)
        num_F2 = 2 * (1 + k * k_psi - nu * (k + 1) * (k_psi + 1))
        den_F2 = (k - 1) * (k + k_psi)
        F2 = num_F2 / den_F2
        F3 = 2 * (1 - nu) * (k + 1) / (k + k_psi)

        term_parentesi = F1 + F2 * ((1/Rp)**(k - 1)) + F3 * (Rp**(k_psi + 1))
        u_r_R = lam_e * (sig_0 / (2 * G)) * term_parentesi
        
    
    else: # lam > lam_a

        num_Ra = (1 - 2*nu) * (k + 1) * lam_e
        den_Ra = ((1 - nu) * k - nu) * ((k + 1) * lam_e - (k - 1) * lam)
        Ra = (num_Ra / den_Ra) ** (1 / (k - 1))

        F1 = -(1 - 2*nu) * (k + 1) / (k - 1)
        num_F2 = 2 * (1 + k * k_psi - nu * (k + 1) * (k_psi + 1))
        den_F2 = (k - 1) * (k + k_psi)
        F2 = num_F2 / den_F2
        F3 = 2 * (1 - nu) * (k + 1) / (k + k_psi)

        A1 = - ((1 - 2*nu) / (1 + nu)) * ((k + 1) / (k - 1)) * ((2*k_psi + 1) / (k_psi + 1)) # da verificare l'ultima tonda perche cambiata rispetto le formule
        
        # A2
        num_A2 = 2 * (1 + 2*k*k_psi - 2*nu*(k + k_psi + k*k_psi))
        den_A2 = (1 + nu) * (k - 1) * (k + k_psi)
        A2 = num_A2 / den_A2
        
        # A3
        term1_A3 = (F1 - A1) * (Ra/Rp)**(k_psi + 1)
        term2_A3 = (F2 - A2) * (Ra/Rp)**(k + k_psi)
        A3 = term1_A3 + term2_A3 + F3

        term_parentesi = A1 + A2 * ((1/Rp)**(k - 1)) + A3 * (Rp**(k_psi + 1))
        u_r_R = lam_e * (sig_0 / (2 * G)) * term_parentesi

    return u_r_R , eta , lam, lam_e, lam_a



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

col1, col2 = st.columns(2)
with col1:
    gamma = st.number_input("Unit weight γ (kN/m³)", min_value=0.0, value=25.0, step=0.5)
with col2:
    h = st.number_input("Overburden H (m)", min_value=0.0, value=100.0, step=10.0)

col3, col4, col5 = st.columns(3)
with col3:
    R = st.number_input("Tunnel radius R (m)", min_value=0.0, value=5.0, step=0.5)
with col4:
    E = st.number_input("Elastic modulus E (GPa)", min_value=0.0, value=5.0, step=0.5)
with col5:
    nu = st.number_input("Poisson ratio ν (-)", min_value=0.0, value=0.25, step=0.05)

col6, col7 = st.columns(2)
with col6:
    phi_deg = st.number_input("Friction angle φ (°)", min_value=0.0, value=30.0, step=1.0)
with col7:
    coh = st.number_input("Cohesion c (MPa)", min_value=0.0, value=1.0, step=0.1)

st.divider()
# --------------------------------------------------
# CONVERT φ
# --------------------------------------------------

phi_rad = np.radians(phi_deg)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("Predict"):

    y_true, eta, lam, lam_e, lam_a = compute_d_r_R_new(E*1e9, nu, gamma*1e3, h, R, phi_rad, coh*1e6, psi, p)
    X = np.array([[nu, phi_rad, eta]])
    X_scaled = scaler_X.transform(X)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_pred = y_pred * R * (gamma*1e3) * h / (E*1e9)


    error_percent = 100 * abs(y_pred[0][0] - y_true) / abs(y_true)


    # 1. Mostriamo prima i parametri derivati (come eta)
    st.markdown("### Derived Parameters")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("η (Plastic Radius Ratio)", f"{eta:.2f}")
    with c2:
        st.metric("λ (Current)", f"{lam:.2f}")
    with c3:
        st.metric("λₑ (Elastic)", f"{lam_e:.2f}")
    with c4:
        st.metric("λₐ (Edge)", f"{lam_a:.2f}")

    # Messaggio di stato del regime
    if lam <= lam_e:
        st.info("State: **ELASTIC**")
    elif lam <= lam_a:
        st.success("State: **PLASTIC (Face Mode)**")
    else:
        st.warning("State: **PLASTIC (Edge Mode)**")

    st.divider()

    st.markdown("### Validation against analytical solution")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("ANN prediction", f"{y_pred[0][0]*1e3:.2f} (mm)")

    with col2:
        st.metric("Analytical value", f"{y_true*1e3:.2f} (mm)")

    with col3:
        st.metric("Error (%)", f"{error_percent:.2f}")


