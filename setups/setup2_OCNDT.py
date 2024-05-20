import numpy as np
import math
from geometry import *

outfilename = 'Setup2_OCNDT'

Lx = 1*150e-3 # Block width
Lz = 1*150e-3 # Block height
Lt = 150e-6 # Simulation time

dx = 300e-6 # 300 x-axis step (1e-4) or 5e-5 for water
# dz = 1500e-6 # 300 z-axis step (1e-4) or 5e-5 for water

dt = 25e-9 # 25 Sampling time 5e-9 or 8e-9 for water

Nx = round(Lx/dx)
Nz = round(Lz/dx)
Nt = round(Lt/dt)

PML_size = 20

ad = dx/dt # Adimensionality constant

# Sound speeds
cMat = {}
cMat['water'] = 1450/ad
cMat['acrylic'] = 2730/ad
cMat['steel'] = 5800/ad # 5490

vmin = 0.9*min(cMat.values())
vmax = 1.1*max(cMat.values())

f0 = 1e6 # 5 Transducer central frequency
t0 = 5e-6 # 1 Pulse time
grad_scale = 1e-1

# Source signal
t = np.linspace(0, Lt-dt, Nt)
bwp = .9  # Bandwidth in percentage
bw = bwp*f0
t0 = 1/f0 + 1/bw
# t0 = 1/f0
alpha = -(math.pi*bw/2)**2/math.log(math.sqrt(2)/2)


def signal(tt0=t0):
    ss = np.exp(-alpha*(t-tt0)**2)*np.cos(2*np.pi*f0*(t-tt0))
    return ss


# Source location
xzs = np.full((Nz, Nx), False)

zs = 20e-3 # 20e-3
width_s = 100e-3 # 100e-3
n_elem_s = 3 # Number of sources each array

p1a = int(zs/dx)
p2a = np.round((Lx/2 + np.linspace(-width_s/2, width_s/2, n_elem_s))/dx).astype(int)
xzs[p1a, p2a] = True
xzs[-p1a, p2a] = True

Ns = int(np.sum(xzs))
s = np.zeros((Ns,Nt))
for i in range(Ns):
    s[i,:] = signal((1)*t0) # zero delay

# Measurement points
xzm = np.full((Nz, Nx), False)
zt = 20e-3
width_t = 100e-3
n_elem = 64

p1a = int(zt/dx)
p2a = np.round((Lx/2 + np.linspace(-width_t/2, width_t/2, n_elem-1))/dx).astype(int)
xzm[p1a, p2a] = True
xzm[-p1a, p2a] = True

Nm = int(np.sum(xzm))

# Gradient mask
gmask = np.full((Nz,Nx),0).astype(int)
gmask_coord = np.asarray([[0.25*Nx, 0.75*Nx, 0.75*Nx, 0.25*Nx, 0.25*Nx],[0.25*Nz, 0.25*Nz, 0.75*Nz, 0.75*Nz, 0.25*Nz]])
draw_rectangle(gmask, 0.5*Nz, 0.5*Nx, centro=(Nx//2, Nz//2))

# Velocity field
params_real = cMat['water']*np.ones((Nz, Nx))
draw_rectangle(params_real, 0.3*Nz, 0.3*Nx, value=cMat['steel'])

estimativa = cMat['water']*np.ones((Nz, Nx))


