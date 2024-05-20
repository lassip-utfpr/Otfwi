import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal as sig


def add_array(xzm, center, n_elements, pitch, dx, dz, angle=0, elem_size=1):
    idx = np.zeros((2, n_elements), dtype=int)
    idx[0, :] = center[0]/dx
    length_array = (n_elements - 1)*pitch/dx + elem_size
    idx[1, :] = np.round(center[1]/dx + np.arange(-length_array/2, length_array/2, pitch/dx)).astype(int)
    xzm[idx[0,:], idx[1,:]] = True

    return idx


def pulse(t, f0):
    bwp = .9  # Bandwidth in percentage
    bw = bwp*f0
    # t0 = 1/f0 + 1/bw
    t0 = 2/f0
    alpha = -(math.pi*bw/2)**2/math.log(math.sqrt(2)/2)
    s = np.exp(-alpha*(t-t0)**2)*np.cos(2*np.pi*f0*(t-t0))
    return s


def add_noise(u, db):
    s = np.sqrt(np.mean(u**2)*10**(-db/10))
    return u + s*np.random.randn(*u.shape)


def plotaModelo(model, ad, dx, xzm, xzs, vmin=None, vmax=None, title=None, colormap='jet', tick=25, scale=1000, gmask_coord=None):
    fig = plt.figure(figsize=[4.8,4.20])
    if vmin is None:
        vmin = np.min(model)
    if vmax is None:
        vmax = np.max(model)
    Lx = model.shape[1]*dx*scale
    Lz = model.shape[0]*dx*scale
    plt.imshow((model*ad), vmin=vmin*ad, vmax=vmax*ad, cmap=colormap, interpolation='nearest', extent=[0, model.shape[1]*dx*scale, model.shape[0]*dx*scale, 0])
    plt.colorbar()
    plt.plot(np.where(xzm)[1]*dx*scale, np.where(xzm)[0]*dx*scale, 'y.', label='RX')
    plt.plot(np.where(xzs)[1]*dx*scale, np.where(xzs)[0]*dx*scale, 'mo', label='TX')
    plt.xticks(np.arange(0, Lx+1, tick))
    plt.yticks(np.arange(0, Lz+1, tick))
    plt.legend(loc=6,borderpad=0.1,borderaxespad=0.1, handletextpad=0.1)
    if title is not None:
        plt.title(title)
    plt.xlabel('[mm]')
    plt.ylabel('[mm]')
    plt.grid()
    if gmask_coord is not None:
        plt.plot(scale*dx*gmask_coord[0], scale*dx*gmask_coord[1], linestyle='--', linewidth=3, color='lime')
    return fig


def envelope(v):
    hilb = np.abs(sig.hilbert(v))
    return hilb

