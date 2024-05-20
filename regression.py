import os
import numpy as np
import time
import matplotlib
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, OptimizeResult
import pickle

from cuda_interface_pml import Cuda_interface_PML
import adjoint_functions as adj
from utils import plotaModelo, add_noise
from geometry import *
import matplotlib.gridspec as gridspec

matplotlib.use('TkAgg')
colormap = 'jet'


# CHANGE THIS LINE TO SELECT DIFFERENT TEST SETUPS
from setups.setup1_OCNDT import *

print(cMat.values())
courant = max(cMat.values())
if courant > 1:
    raise ValueError("Courant error")
else:
    print("Courant OK: ",courant)

plotaModelo(params_real, ad, dx, xzm, xzs, title='', vmin=vmin, vmax=vmax, tick=25, gmask_coord=gmask_coord)
plt.title('Observed specimen')
plt.show()

cp = Cuda_interface_PML(Nx, Nz, Nt, PML_size, params_real, Ns, xzs, Nm, xzm, s, multishot=True)
start = time.time()
cp.simulate()
end = time.time()
print('Simulation time: ' + str(end-start) + "s")
sinal_observado = cp.recording.copy()

# cp.init_regression(add_noise(cp.recording, -20), adj.wasserstein,  prec_deriv=4)
cp.init_regression(cp.recording, adj.wasserstein,  prec_deriv=4)


def objfun(params):
    cp.calc_grad(c=params.flatten())
    grad = grad_scale*(gmask*cp.grad).flatten()
    print([1e3*cp.mse, np.max(grad), np.min(grad)])
    return 1e3*cp.mse, grad


model_misfit = np.zeros(300)
n = 0


def callback(params):
    global n
    fig, ax = plt.subplots(1, 1, figsize=(10, 9), layout="constrained")
    axs = [ax]

    fig.suptitle('Iteração ' + '{:0>4}'.format(n), fontsize=24)

    im1 = axs[0].imshow((params * ad).reshape((Nz, Nx)), vmin=vmin*ad, vmax=vmax*ad, cmap=colormap, interpolation='nearest', extent=[0, 1000 * Lx, 1000 * Lz, 0])
    axs[0].set_xlabel('[mm]', fontsize=16)
    axs[0].set_ylabel('[mm]', fontsize=16)
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    plt.xticks(np.arange(0, Lx*1000+1, 25))
    plt.yticks(np.arange(0, Lz*1000+1, 25))
    axs[0].set_title('Mapa de velocidades estimado [m/s]', fontsize=18)

    cbar = fig.colorbar(im1, ax=axs, shrink=0.8, location='right')
    cbar.ax.tick_params(labelsize=14)

    fig.savefig('outreg/reg' + '{:0>4}'.format(n) + '.png')
    plt.close('all')
    misfit = (np.linalg.norm(params.reshape(Nz, Nx) - params_real)**2)/(Nz*Nx)
    model_misfit[n] = misfit
    n += 1


os.system('rm -f outreg/*.png')
bnds = Bounds(vmin, vmax)
opt = {'disp':True, 'gtol': 1e-8, 'ftol': 1e-8, 'maxiter': 200, 'maxcor': 20, 'maxls': 10}
mthd = 'L-BFGS-B'

start = time.time()
r = minimize(objfun, estimativa.flatten(), jac=True, method=mthd, bounds=bnds, options=opt, callback=callback)
print((time.time()-start))

# calculate L2 error L2
cp.simulate(c=r.x.reshape((Nz, Nx)))
mseL2, _ = adj.L2(sinal_observado, cp.recording)


with open('output/regression_' + outfilename + '.pickle', 'wb') as f:
    pickle.dump([cp.recording, r.x.reshape((Nz,Nx)), params_real, xzm, xzs, vmin, vmax, r.fun, mseL2, model_misfit], f)


os.system('ffmpeg -y -framerate 5 -i outreg/reg%04d.png output/outreg_' + outfilename + '.mp4 -f mp4 -vcodec libx264 -crf 0 -nostats -loglevel quiet')

plotaModelo(r.x.reshape((Nz,Nx)), ad, dx, xzm, xzs, vmin=vmin, vmax=vmax, tick=10, gmask_coord=gmask_coord)
plt.show()
print(1)



