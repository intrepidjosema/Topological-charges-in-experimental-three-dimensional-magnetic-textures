import discretisedfield as df
import numpy as np

# 1. Parámetros de la malla (coincidentes con tu script de MuMax3)
N = 124
cellsize = 2e-9  # 1 nm
L = N * cellsize  # 64 nm

p1 = (-L / 2, -L / 2, -L / 2)
p2 = (L / 2, L / 2, L / 2)
mesh = df.Mesh(p1=p1, p2=p2, n=(N, N, N))

# 2. Definición de parámetros físicos
R = 70e-9  # Radio de la esfera (60 nm de diámetro)
gamma = 1.823  # Helicidad (104.5 grados)

# 3. Función del mapa de magnetización
def bloch_point_ansatz(pos):
    x, y, z = pos
    r = np.sqrt(x**2 + y**2 + z**2)

    # Fuera de la esfera la magnetización es nula
    if r > R or r == 0:
        return (0, 0, 0)

    # Ansatz analítico (Erizo con helicidad)
    mx = (x * np.cos(gamma) - y * np.sin(gamma)) / r
    my = (y * np.cos(gamma) + x * np.sin(gamma)) / r
    mz = z / r

    return (mx, my, mz)


field_raw = df.Field(mesh, nvdim=3, value=bloch_point_ansatz)
m_data = field_raw.array  

# 4. Máscara elipsoidal 3D booleana/float (32, 32, 32)
x_1d = np.linspace(p1[0] + cellsize / 2, p2[0] - cellsize / 2, N)
y_1d = np.linspace(p1[1] + cellsize / 2, p2[1] - cellsize / 2, N)
z_1d = np.linspace(p1[2] + cellsize / 2, p2[2] - cellsize / 2, N)
x, y, z = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")


Rx, Ry, Rz = R, R, R * 1.0  # Elongación elipsoidal (APAGADA)
ellipsoid_mask = (
    (x / Rx) ** 2 + (y / Ry) ** 2 + (z / Rz) ** 2 <= 1.0
).astype(np.float32)

# Máscara 4D limpia usando np.expand_dims para evitar duplicar dimensiones (32, 32, 32, 1)
mask_4d = np.expand_dims(ellipsoid_mask, axis=-1)


# ==========================================
# A) BLOCH IDEAL CON ELIPSE (Forma: 32, 32, 32, 3)
# ==========================================
m_ideal_elipse = m_data * mask_4d

field_ideal = df.Field(mesh, nvdim=3, value=m_ideal_elipse)
field_ideal.to_file("BlochPoint_R70nm_Ideal_Sphere.ovf")
print("-> BlochPoint_R70nm_Ideal_Sphere.ovf generated")

# Quitamos la dimensión extra que usamos para multiplicar (forma: 124, 124, 124)
mask_scalar = ellipsoid_mask.astype(np.float32)

mask_field = df.Field(mesh, nvdim=1, value=mask_scalar)
mask_field.to_file("BlochPoint_R70nm_Ideal_Sphere_mask.ovf")
print("-> BlochPoint_R70nm_Ideal_Sphere_mask.ovf generated")