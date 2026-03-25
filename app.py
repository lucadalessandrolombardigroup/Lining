import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# --------------------------------------------------
# CACHE MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model_and_scalers():

    model = tf.keras.models.load_model("ANN_model/ann_surrogate1.keras")
    scaler_X = joblib.load("ANN_model/scaler_X1.pkl")
    scaler_y = joblib.load("ANN_model/scaler_y1.pkl")

    return model, scaler_X, scaler_y


# --------------------------------------------------
# ANALYTICAL MODEL (for validation)
# --------------------------------------------------

psi = 0

def compute_d_r_R(Erm, nu, gamma, h, phi_rad, coh, psi, lam):
    
    G = Erm / (2 * (1 + nu))
    k = (1 + np.sin(phi_rad)) / (1 - np.sin(phi_rad))
    k_psi = (1 + np.sin(psi)) / (1 - np.sin(psi))
    
    sig_0 = gamma*h
    
    #lam = 1- p/sig_0
    p = (1 - lam) * sig_0
    sig_c = (2 * coh * np.cos(phi_rad)) / (1 - np.sin(phi_rad))
    lam_e = ((k - 1) * sig_0 + sig_c) / ((k + 1) * sig_0)


    den_lam_a = k - nu * (k + 1)
    if den_lam_a != 0:
        lam_a = lam_e * ((1 - nu) * (k + 1)) / den_lam_a
    else:
        lam_a = np.nan

    Rp = ((2*lam_e)/((k+1)*lam_e - (k-1)*lam)) ** (1 / (k - 1)) # sarebbe Rp /R
    eta = Rp - 1
        
    if lam <= lam_e:
        u_r_R =  (1 + nu) / Erm * (sig_0 - p)
        eta = 0.0
    
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

    return u_r_R , eta , p, lam_e, lam_a

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
    gamma = st.number_input("Unit weight γ (kN/m³)", min_value=0.0, value=0.0, step=1.0)
with col2:
    h = st.number_input("Overburden H (m)", min_value=0.0, value=0.0, step=10.0)

col3, col4, col5 = st.columns(3)
with col3:
    R = st.number_input("Tunnel radius R (m)", min_value=0.0, value=0.0, step=0.5)
with col4:
    E = st.number_input("Elastic modulus E (GPa)", min_value=0.0, value=0.0, step=0.5)
with col5:
    nu = st.number_input("Poisson ratio ν (-)", min_value=0.0, value=0.0, step=0.05)

col6, col7 = st.columns(2)
with col6:
    phi_deg = st.number_input("Friction angle φ (°)", min_value=0.0, value=0.0, step=5.0)
with col7:
    coh = st.number_input("Cohesion c (kPa)", min_value=0.0, value=0.0, step=0.5)

st.divider()
# --------------------------------------------------
# CONVERT φ
# --------------------------------------------------

phi_rad = np.radians(phi_deg)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------


import matplotlib.pyplot as plt

if st.button("Predict"):

    if R <= 0 or E <= 0 or gamma <= 0 or h <= 0 or coh <= 0:
        st.error("⚠️ Please ensure that geometry and material parameters are greater than zero.")
    else:
        sig_0 = gamma * 1e3 * h   # Pa
        Erm = E * 1e9             # Pa
        c = coh * 1e3             # Pa

        lam_list = np.linspace(0.0, 1.0, 101)

        p_vals = []
        u_analytical_vals = []
        eta_vals = []

        for lam in lam_list:
            # Analytical solution
            y_true, eta, p, lam_e, lam_a = compute_d_r_R(
                Erm, nu, gamma * 1e3, h, phi_rad, c, psi, lam
            )

            u_true = y_true * R   

            p_vals.append(p / 1e3)              
            u_analytical_vals.append(u_true * 1e2)  # cm
            eta_vals.append(eta)


        lam_top = 1.0

        y_true_top, eta_top, p_top, lam_e_top, lam_a_top = compute_d_r_R(
            Erm, nu, gamma * 1e3, h, phi_rad, c, psi, lam_top
        )

        u_true_top = y_true_top * R   # m

        # ANN input for the old model: [nu, phi_rad, eta]
        X = np.array([[nu, phi_rad, eta_top]])
        X_scaled = scaler_X.transform(X)
        y_pred_scaled = model.predict(X_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        # assuming ANN output = (u/R) * (E/sig0)
        u_pred_top = y_pred.item() * R * sig_0 / Erm   # m

        error_percent = 100 * abs(u_pred_top - u_true_top) / abs(u_true_top)        

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(p_vals, u_analytical_vals)

        ax.scatter(
            [p_top / 1e3],                # kPa
            [u_pred_top * 1e2],           # cm
            marker='o',
            label="ANN prediction at p = 0",
            color = 'green',
            facecolors='none',
            zorder=5
        )


        ax.set_xlabel("Radial pressure (kPa)")
        ax.set_ylabel("Radial displacement (cm)")
        ax.set_title("Ground Reaction Curve")
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.legend()
        st.pyplot(fig)


        # 1. Mostriamo prima i parametri derivati (come eta)
        st.markdown("### Derived parameters (p=0)")
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
        elif lam_e < lam <= lam_a:
            st.success("State: **PLASTIC (Face Mode)**")
        else:
            st.warning("State: **PLASTIC (Edge Mode)**")

        st.divider()

        st.markdown("### Validation against analytical solution (p=0)")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("ANN prediction", f"{u_pred_top*1e2:.2f} (cm)")

        with col2:
            st.metric("Analytical value", f"{u_true_top*1e2:.2f} (cm)")

        with col3:
            st.metric("Error (%)", f"{error_percent:.2f}")

