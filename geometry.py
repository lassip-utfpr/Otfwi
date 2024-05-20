import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.draw import polygon
from scipy.ndimage import correlate


def star_idx(shape, altura, centro):
    externalAngles = np.deg2rad(np.arange(90,90-360,-360/5) - 36)
    internalAngles = np.deg2rad(np.arange(90,90-360,-360/5))

    xx = np.zeros(10)
    zz = np.zeros(10)
    centro = np.asarray(centro)
    for i in range(5):
        xx[2*i] = np.sin(np.deg2rad(22))*altura*np.sin(internalAngles[i])+centro[0]
        zz[2*i] = np.sin(np.deg2rad(22))*altura*np.cos(internalAngles[i])+centro[1]
        xx[2*i+1] = altura*np.sin(externalAngles[i])+centro[0]
        zz[2*i+1] = altura*np.cos(externalAngles[i])+centro[1]

    rr, cc = polygon(xx, zz, shape)
    return rr, cc


def draw_rectangle(target, altura, largura, centro=None, value=1):
    shape = target.shape
    if centro is None:
        centro = [int(shape[0]/2), int(shape[1]/2)]

    origem = np.asarray([0, 0])
    origem[0] = int(centro[0] - largura/2)
    origem[1] = int(centro[1] - altura/2)
    canto = np.asarray([0, 0])
    canto[0] = int(centro[0] + largura/2)
    canto[1] = int(centro[1] + altura/2)
    target[origem[1]:canto[1], origem[0]:canto[0]] = value


def draw_circle(target, diametro, centro=None, value=1):
    shape = target.shape
    if centro is None:
        centro = [int(shape[0]/2), int(shape[1]/2)]
    r = int(round(diametro//2))
    x = int(round(centro[0]))
    z = int(round(centro[1]))
    v_z, v_x = np.ogrid[-z:shape[1]-z, -x:shape[0]-x]
    vmask = v_x**2 + v_z**2 <= r**2
    target[vmask] = value

