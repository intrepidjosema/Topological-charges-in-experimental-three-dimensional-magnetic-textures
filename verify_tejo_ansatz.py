import discretisedfield as df
import numpy as np
import matplotlib.pyplot as plt

def verify_tejo_ansatz_from_ovf(ovf_filename, mask_ovf_filename=None):
    # 1. Cargar el campo OVF
    field = df.Field.from_file(ovf_filename)
    m_data = field.array  # Dimensiones nativas: (Nx, Ny, Nz, 3)
    
    mx = m_data[..., 0]
    my = m_data[..., 1]
    mz = m_data[..., 2]
    
    # 2. Cargar o generar la máscara geométrica
    if mask_ovf_filename:
        mask_field = df.Field.from_file(mask_ovf_filename)
        mask = mask_field.array[..., 0] > 0.5
    else:
        norm_m = np.sqrt(mx**2 + my**2 + mz**2)
        mask = norm_m > 0.1
        
    Nx, Ny, Nz = mx.shape
    dx, dy, dz = field.mesh.cell
    print(f"OVF cargado: {Nx}x{Ny}x{Nz} celdas | dx={dx*1e9:.2f} nm, dy={dy*1e9:.2f} nm, dz={dz*1e9:.2f} nm")

    # 3. Malla de coordenadas relativas al centro (origen en el defecto)
    xc, yc, zc = (Nx - 1) / 2.0, (Ny - 1) / 2.0, (Nz - 1) / 2.0
    ix = np.arange(Nx) - xc
    iy = np.arange(Ny) - yc
    iz = np.arange(Nz) - zc
    X, Y, Z = np.meshgrid(ix * dx, iy * dy, iz * dz, indexing='ij')

    R_sp = np.sqrt(X**2 + Y**2 + Z**2)
    R_sp[R_sp == 0] = 1e-15

    Theta = np.arccos(np.clip(Z / R_sp, -1.0, 1.0))
    Phi = np.arctan2(Y, X)

    sin_T, cos_T = np.sin(Theta), np.cos(Theta)
    sin_P, cos_P = np.sin(Phi), np.cos(Phi)

    # 4. Proyección a la base esférica (m_r, m_theta, m_phi)
    m_r = sin_T * cos_P * mx + sin_T * sin_P * my + cos_T * mz
    m_theta = cos_T * cos_P * mx + cos_T * sin_P * my - sin_T * mz
    m_phi = -sin_P * mx + cos_P * my

    # 5. Extracción de gamma evitando división por cero en los polos (theta ~ 0, pi)
    sin_T_safe = np.where(sin_T > 0.15, sin_T, np.nan)
    num = m_phi / sin_T_safe
    den = (m_r - cos_T**2) / (sin_T_safe**2)
    gamma_map = np.arctan2(num, den)

    # 6. Comprobación algebraica de consistencia interna (m_r vs m_theta)
    cos_gamma_r = den
    cos_gamma_theta = 1.0 + m_theta / (sin_T_safe * cos_T + 1e-12)
    discrepancy = np.nanmean(np.abs(cos_gamma_r - cos_gamma_theta)[mask])
    print(f"Discrepancia algebraica interna (|cos gamma_r - cos gamma_theta|): {discrepancy:.4f}")

    # 7. Comprobación de independencia azimutal en el ecuador (z = 0)
    mid_z = int(round(zc))
    gamma_eq = gamma_map[:, :, mid_z]
    mask_eq = mask[:, :, mid_z]
    
    std_phi_eq = np.nanstd(gamma_eq[mask_eq])
    print(f"Desviación azimutal en el ecuador (sigma_phi): {np.degrees(std_phi_eq):.2f}°")

    # 8. Gráfica 2D del corte meridional gamma(y, z) a x = 0 (comparar con Fig. 2 de Tejo et al.)
    mid_x = int(round(xc))
    gamma_slice = np.degrees(gamma_map[mid_x, :, :])
    gamma_slice[~mask[mid_x, :, :]] = np.nan

    extent = [iy[0] * dy * 1e9, iy[-1] * dy * 1e9, iz[0] * dz * 1e9, iz[-1] * dz * 1e9]

    plt.figure(figsize=(6, 5))
    im = plt.imshow(gamma_slice.T, origin='lower', extent=extent, cmap='jet', vmin=70, vmax=112)
    plt.colorbar(im, label=r'$\gamma(y, z)$ (grados)')
    plt.xlabel('y (nm)')
    plt.ylabel('z (nm)')
    plt.title(r'Perfil $\gamma(y, z)$ relajado desde OVF')
    plt.tight_layout()
    plt.show()

# Ejecución directa pasando tus archivos OVF:
# verify_tejo_ansatz_from_ovf("controlled_blochpoints_m.ovf", "controlled_blochpoints_m_mask.ovf")