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

csvfile = open('dataset.csv', 'w', newline='')
writer = csv.writer(csvfile, delimiter=',')
writer.writerow(['Erm_kPa', 'nu', 'sig_0_kPa', 'phi_rad', 'eta', 'd_r_R_permille'])

p = 0  # internal pressure

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


for sig_0 in sig_0_list:

    nrows = 3
    ncols = 3

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                             sharex=True, sharey=True,
                             figsize=(12, 10))

    for i, nu in enumerate(nu_list):
        for j, Erm in enumerate(Erm_list):

            if i < nrows and j < ncols:
                ax = axes[i, j]

                for phi in phi_list:
                    phi_rad = np.radians(phi)
                    d_series = []

                    for eta in eta_list:
                        d_r_R = compute_d_r_R_permille(
                            Erm, nu, sig_0, phi_rad, eta, psi, p
                        )

                        writer.writerow([Erm, nu, sig_0, phi_rad, eta, d_r_R])
                        d_series.append(d_r_R*Erm/sig_0)

                    ax.plot(eta_list, d_series, marker='o', label=f'{phi}°')

                # Titles only on top row
                if i == 0:
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

csvfile.close()