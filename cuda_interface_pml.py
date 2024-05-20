import os
import time

import numpy as np
import ctypes
from ctypes import c_float, c_int, CDLL, CFUNCTYPE, POINTER


lib2_path = 'simulator/bin/cuda_regression_PML_single.so'
try:
    clib = CDLL(os.path.abspath(lib2_path))
except:
    print('Error importing library')

CB_FTYPE_ADJ = CFUNCTYPE(c_float, POINTER(c_float), POINTER(c_float))

cuda_init_sim = clib.init_memory_sim
cuda_init_sim.restype = None

cuda_init_reg = clib.init_memory_reg
cuda_init_reg.restype = None

cuda_set_cquad = clib.setCquad
cuda_set_cquad.restype = None
#
cuda_simulate_c = clib.cuda_simulate
cuda_simulate_c.restype = None
#
cuda_grad_c = clib.cuda_grad_ext
cuda_grad_c.restype = None


class Cuda_interface_PML:
    def __init__(self, X, Z, T, PML_size, c, n_source, pos_source, n_sensor, pos_sensor, source, initial=None, multishot=False):
        self.c_float_p = ctypes.POINTER(ctypes.c_float)
        self.c_int_p = ctypes.POINTER(ctypes.c_int)

        # TODO: typecheck e sizecheck dos parametros

        self.pos_sensor = np.where(pos_sensor)
        self.pos_source = np.where(pos_source)

        self.X = X
        self.Z = Z
        self.T = T
        self.PML_size = PML_size
        self.cquad = (c**2).astype(np.single)
        self.n_source = n_source
        self.pos_source_x = self.pos_source[1]
        self.pos_source_z = self.pos_source[0]
        self.n_sensor = n_sensor
        self.pos_sensor_x = self.pos_sensor[1]
        self.pos_sensor_z = self.pos_sensor[0]
        self.source = source.astype(np.single)

        self.d_x, self.d_z = self._calcPML()
        # self.d_x, self.d_z = np.zeros_like(self.cquad).astype(np.single), np.zeros_like(self.cquad).astype(np.single)
        self.constvec = np.asarray([1,1,1]).astype(np.single)
        self._pos_revert = None
        self._pos_revert_x = None
        self._pos_revert_z = None
        self._tammaskrev = 0

        self.grad = np.zeros((Z, X)).astype(np.single)
        self.mse = np.zeros(1)

        FuncType = ctypes.CFUNCTYPE(None, POINTER(c_float), POINTER(c_float))
        self.adj_wrapper_ptr = ctypes.cast(FuncType(self.adj_wrapper), ctypes.c_void_p)
        self._adj_fun = None

        self._multishot = multishot
        if multishot:
            self.recording = np.zeros((n_source, n_sensor, T)).astype(np.single)
        else:
            self.recording = np.zeros((n_sensor, T)).astype(np.single)

        self.idx_source = -1

        self.cquadptr = self.cquad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if initial is None:
            self.initial = np.zeros((Z, X, 2))
        else:
            self.initial = initial

        cq = self.cquad.flatten()
        init = self.initial.flatten()
        src = self.source.flatten()
        ddx = self.d_x.flatten()
        ddz = self.d_z.flatten()

        X_int = c_int(X)
        Z_int = (c_int)(Z)
        T_int = (c_int)(T)
        cquad_arr = (c_float * len(cq))(*cq)
        initial_arr = (c_float * len(init))(*init)
        d_x_arr = (c_float * len(ddx))(*ddx)
        d_z_arr = (c_float * len(ddx))(*ddz)
        constvec_arr = (c_float * len(self.constvec))(*self.constvec)

        self.observed = None
        self.gradptr = None

        n_source_int = c_int(self.n_source)
        source_x_arr = (c_int * len(self.pos_source_x))(*self.pos_source_x)
        source_z_arr = (c_int * len(self.pos_source_z))(*self.pos_source_z)
        source_arr = (c_float * len(src))(*src)

        n_sensor_int = c_int(self.n_sensor)
        sensor_x_arr = (c_int * len(self.pos_sensor_x))(*self.pos_sensor_x)
        sensor_z_arr = (c_int * len(self.pos_sensor_z))(*self.pos_sensor_z)

        self.recptr = np.asarray((0)).ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))
        cuda_init_sim(X_int, Z_int, T_int, cquad_arr, constvec_arr, d_x_arr, d_z_arr, n_source_int, source_x_arr, source_z_arr, n_sensor_int, sensor_x_arr, sensor_z_arr, source_arr, initial_arr, self.recptr)

    def _calcPML(self):
        R = 1e-5
        Vmax = 0.5
        Lp = self.PML_size * 1

        d0 = 3 * Vmax * np.log(1 / R) / (2 * Lp ** 3)
        x = np.linspace(0, Lp, self.PML_size)
        damp_profile = d0 * x[:, np.newaxis]**2

        d_x = np.zeros((self.Z, self.X))
        d_z = np.zeros((self.Z, self.X))
        d_z[:self.PML_size, :] = damp_profile[-1::-1]
        d_z[-self.PML_size:, :] = damp_profile
        d_x[:, -self.PML_size:] = damp_profile.T
        d_x[:, :self.PML_size] = damp_profile[-1::-1].T

        return d_x, d_z

    def init_regression(self, observed, adj_fun, prec_deriv=8):
        margem = prec_deriv+1
        maskrevert = np.full((self.Z, self.X), False)
        maskrevert[self.PML_size:self.PML_size + margem, self.PML_size:-self.PML_size] = True
        maskrevert[-self.PML_size - margem:-self.PML_size, self.PML_size:-self.PML_size] = True
        maskrevert[self.PML_size:-self.PML_size, self.PML_size:self.PML_size + margem] = True
        maskrevert[self.PML_size:-self.PML_size, -self.PML_size - margem:-self.PML_size] = True
        maskrevert[self.pos_source] = True
        # maskrevert[self.pos_sensor] = True

        self._pos_revert = np.where(maskrevert)
        self._pos_revert_x = self._pos_revert[1]
        self._pos_revert_z = self._pos_revert[0]

        self._tammaskrev = np.sum(maskrevert)

        revert_x_arr = (c_int * len(self._pos_revert_x))(*self._pos_revert_x)
        revert_z_arr = (c_int * len(self._pos_revert_z))(*self._pos_revert_z)
        n_revert_int = c_int(self._tammaskrev)

        self.observed = observed
        observed_arr = (c_float * len(observed.flatten()))(*observed.flatten())
        self.gradptr = np.asarray((0)).ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))  # ponteiro do vetor gradiente
        self._adj_fun = adj_fun

        cuda_init_reg(observed_arr, self.gradptr, revert_x_arr, revert_z_arr, n_revert_int)

    def _set_c_quad(self, cquad):
        self.cquad = cquad.astype(np.single)
        cquadptr = self.cquad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        cuda_set_cquad(cquadptr)

    def _set_source(self, n_source, pos_source, source):
        self.pos_source = np.where(pos_source)

        self.source = source.astype(np.single)
        self.n_source = n_source
        self.pos_source_x = pos_source[1]
        self.pos_source_z = pos_source[0]

        self.sourceptr = self.source.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        src = self.source.flatten()

        n_source_int = c_int(self.n_source)
        source_x_arr = (c_int * len(self.pos_source_x))(*self.pos_source_x)
        source_z_arr = (c_int * len(self.pos_source_z))(*self.pos_source_z)
        source_arr = (c_float * len(src))(*src)

        clib.set_source(n_source_int, source_x_arr, source_z_arr, source_arr)

    def simulate(self, output=False, c=None, idx_source=None):
        out_int = 0
        if output:
            out_int = 1
        if c is not None:
            self._set_c_quad(c ** 2)
        if idx_source is not None:
            self.idx_source = idx_source
        else:
            self.idx_source = -1

        if not self._multishot:
            cuda_simulate_c(out_int, c_int(self.idx_source))
            ctypes.memmove(self.recording.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.recptr[0], self.n_sensor * self.T * ctypes.sizeof(ctypes.c_float))
            self._multishot = False
        else:
            buffer = np.zeros((self.n_sensor, self.T)).astype(np.single)
            for s in range(self.n_source):
                cuda_simulate_c(out_int, c_int(s))
                ctypes.memmove(buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.recptr[0], self.n_sensor * self.T * ctypes.sizeof(ctypes.c_float))
                self.recording[s, :, :] = buffer
            self._multishot = True

    def calc_grad(self, c=None, idx_source=None):
        if c is not None:
            self._set_c_quad(c ** 2)
        if idx_source is not None:
            self.idx_source = idx_source
        else:
            self.idx_source = -1

        if not self._multishot:
            mse_f = (c_float * len(self.mse))(*self.mse)
            cuda_grad_c(mse_f, self.adj_wrapper_ptr, c_int(self.idx_source))
            ctypes.memmove(self.grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.gradptr[0], self.Z * self.X * ctypes.sizeof(ctypes.c_float))
        else:
            self.mse[0] = 0
            mse_acc = 0
            grad_acc = np.zeros((self.Z, self.X)).astype(np.single)
            start = time.time()
            for s in range(self.n_source):
                self.idx_source = s
                mse_f = (c_float * len(self.mse))(*self.mse)
                cuda_grad_c(mse_f, self.adj_wrapper_ptr, c_int(s))
                print('.', end='')
                ctypes.memmove(self.grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.gradptr[0], self.Z * self.X * ctypes.sizeof(ctypes.c_float))
                grad_acc += self.grad
                mse_acc += self.mse[0]
            print(time.time() - start)

            self.grad = grad_acc/self.n_source
            self.mse[0] = mse_acc/self.n_source

    def adj_wrapper(self, sim_c, adjoint_source_c):
        simulated = np.asarray(sim_c[:(self.T*self.n_sensor)]).reshape((self.n_sensor, self.T))
        if self._multishot:
            mse, source = self._adj_fun(simulated, self.observed[self.idx_source,:,:])
        else:
            mse, source = self._adj_fun(simulated, self.observed)

        # hack pra copiar o ndarray pro array em C sem fazer laço
        cast_pointer = ctypes.cast(adjoint_source_c, ctypes.POINTER(c_float*(self.T*self.n_sensor)))[0]
        cast_pointer[:self.T*self.n_sensor] = source.flatten()

        self.mse[0] = mse




