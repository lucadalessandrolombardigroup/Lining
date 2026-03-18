import csv
import numpy as np
import matplotlib.pyplot as plt
import math

# Fixed parameter
R = 5  # [m]
psi = 0  # [rad]

# Variable parameters
sig_0_list = [2.5e3, 5e3, 10e3, 20e3]  # [kPa]
eta_list   = [0, 1, 2, 3, 4, 5]        # [-]
phi_list   = [10, 20, 30, 40, 50]      # [°]
nu_list    = [0.1, 0.2, 0.3]           # [-]
Erm_list   = [1e6, 5e6, 10e6]          # [kPa]  # add more values if you want more subplots

#csvfile = open('dataset.csv', 'w', newline='')
#writer = csv.writer(csvfile, delimiter=',')
#writer.writerow(['Erm_kPa', 'nu', 'sig_0_kPa', 'phi_rad', 'eta', 'd_r_R_permille'])

p = 0  # internal pressure

def compute_d_r_R_permille(Erm, nu, sig_0, phi_rad, eta, psi, p):
    k = (1 + np.sin(phi_rad)) / (1 - np.sin(phi_rad))   

    coh = (2 * sig_0 * np.tan(phi_rad)) / ( 
        (1 + k) * ((eta + 1) ** (k - 1) - 2 / (1 + k))
    )

    # Radial displacement for MC criterion
    Rpl = (2 / (1 + k) * (coh / np.tan(phi_rad) + sig_0) / (coh / np.tan(phi_rad) + p)) ** (1 / (k - 1))  
    print ("Rpl: ", Rpl)
    if Rpl < 1:  # CASO ELASTICO
        d_r_R = 1e3 * (1 + nu) / Erm * (sig_0 - p)  # [permille] OK !
    else: # CASO PLASTICO
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

def compute_d_r_R_carranza(Erm, nu, sig_0, phi_rad, eta, psi, p):
    
    # 1. Costanti di base
    G = Erm / (2 * (1 + nu))
    kp = (1 + np.sin(phi_rad)) / (1 - np.sin(phi_rad))
    k_psi = (1 + np.sin(psi)) / (1 - np.sin(psi))
    
    # 2. Calcolo coesione 
    # Nota: la formula di sigma_c richiede la coesione 'coh'
    coh = (2 * sig_0 * np.tan(phi_rad)) / ( 
        (kp + 1) * ((eta + 1) ** (kp - 1) - 2 / (kp + 1))
    )
    
    # 3. sigma_c e lambda_e dalle immagini
    sig_c = (2 * coh * np.cos(phi_rad)) / (1 - np.sin(phi_rad))
    lam_e = ((kp - 1) * sig_0 + sig_c) / ((kp + 1) * sig_0)
    
    # Denominatore della formula per lambda_a
    den_lam_a = kp - nu * (kp + 1)
    
    # Evitiamo divisioni per zero se il materiale è particolare
    if den_lam_a != 0:
        lam_a = lam_e * ((1 - nu) * (kp + 1)) / den_lam_a
    else:
        lam_a = np.nan
        
    print(f"   [INFO] lam_e: {lam_e:.3f} | lam_a: {lam_a:.3f}")
    
    #r_over_Rp = 1 / (1 + eta)
    Rpl = (2 / (1 + kp) * (coh / np.tan(phi_rad) + sig_0) / (coh / np.tan(phi_rad) + p)) ** (1 / (kp - 1))
    r_over_Rp = 1/Rpl
    
    # 5. Fattori F1, F2, F3
    F1 = -(1 - 2*nu) * (kp + 1) / (kp - 1)
    
    num_F2 = 2 * (1 + kp * k_psi - nu * (kp + 1) * (k_psi + 1))
    den_F2 = (kp - 1) * (kp + k_psi)
    F2 = num_F2 / den_F2
    
    F3 = 2 * (1 - nu) * (kp + 1) / (kp + k_psi)
    
    # 6. Calcolo u/r (spostamento radiale / raggio)
    # u_r_R = lam_e * (sig_0 / (2*G)) * [F1 + F2*(r/Rp)^(kp-1) + F3*(Rp/r)^(k_psi+1)]
    term_parentesi = F1 + F2 * (r_over_Rp**(kp - 1)) + F3 * ((1/r_over_Rp)**(k_psi + 1))
    u_r_R = lam_e * (sig_0 / (2 * G)) * term_parentesi
    
    return u_r_R * 1e3  # In permille come l'altra funzione


for sig_0 in sig_0_list:

    nrows = 3
    ncols = 3

    #fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
    #                         sharex=True, sharey=True,
    #                         figsize=(12, 10))

    for i, nu in enumerate(nu_list):
        for j, Erm in enumerate(Erm_list):

            if i < nrows and j < ncols:
     #           ax = axes[i, j]

                for phi in phi_list:
                    phi_rad = np.radians(phi)
                #    d_series = []

                    #for eta in eta_list:
                    #    d_r_R = compute_d_r_R_permille(
                    #        Erm, nu, sig_0, phi_rad, eta, psi, p
                    #    )

                    for eta in eta_list:
                        # 1. Calcolo con la tua formula originale (che internamente calcola la coesione)
                        # Nota: la tua funzione originale usa p=0 per definire la coesione in funzione di eta
                        d_r_R_originale = compute_d_r_R_permille(
                            Erm, nu, sig_0, phi_rad, eta, psi, p
                        )
                        
                        d_r_R_carranza = compute_d_r_R_carranza(Erm, nu, sig_0, phi_rad, eta, psi, p)

                        # 3. Stampa il confronto
                        diff = abs(d_r_R_originale - d_r_R_carranza)
                        print(f"Eta: {eta} | Orig: {d_r_R_originale:.4f} | Carranza: {d_r_R_carranza:.4f} | Diff: {diff:.2e}")

      #                  writer.writerow([Erm, nu, sig_0, phi_rad, eta, d_r_R])
       #                 d_series.append(d_r_R*Erm/sig_0)

#                    ax.plot(eta_list, d_series, marker='o', label=f'{phi}°')

                # Titles only on top row
                '''if i == 0:
                    ax.set_title(f'Erm = {Erm:.2e} kPa')

                # Y labels only on first column
                if j == 0:
                    ax.set_ylabel(f'nu = {nu}')

                ax.grid(True, linestyle='--', linewidth=0.5)

    # Hide unused axes if fewer than 9 combinations
    for i in range(nrows):
        for j in range(ncols):
            if i >= len(nu_list) or j >= len(Erm_list):
                axes[i, j].axis('off')

    fig.suptitle(f'd_r/R vs eta   (sig_0 = {sig_0:.0f} kPa)', fontsize=14)

    fig.supxlabel('eta [-]')
    fig.supylabel('d_r/R*Erm/sig_0 [permille]')

    # Bottom legend (multi-column)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               title='phi',
               loc='lower center',
               ncol=len(phi_list),
               bbox_to_anchor=(0.5, 0.04))

    plt.tight_layout(rect=[0.05, 0.08, 0.95, 0.93])
    plt.savefig(f'sig_0_{sig_0:.0f}_kPa')

csvfile.close()'''