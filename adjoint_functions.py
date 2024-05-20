import numpy as np
import scipy.interpolate as ip
import scipy.signal as sig
import scipy.integrate as integ
import utils


def L2(sim, obs):
    resid = sim - obs
    mse = np.sum(resid**2)
    return mse, resid


def quad_envelope(sim, obs):
    hilbObs = sig.hilbert(obs)
    hilbSim = sig.hilbert(sim)

    resid = np.abs(hilbSim) - np.abs(hilbObs)
    hilbTemp = np.abs(hilbSim)**2 - np.abs(hilbObs)**2
    adj_source = sim * hilbTemp - np.imag(sig.hilbert(hilbTemp * np.imag(hilbSim)))

    return np.sum(hilbTemp**2), adj_source


def wasserstein(sim, obs):
    # minha versao
    [n_sens, T] = sim.shape
    t = np.linspace(0,1,T)
    dt = t[1] - t[0]
    W2 = 0
    s_W2 = np.zeros((n_sens, T))
    gate_samples = 0
    for s in range(n_sens):

        f_gate = np.concatenate([np.zeros(gate_samples), sim[s, gate_samples:]])
        g_gate = np.concatenate([np.zeros(gate_samples), obs[s, gate_samples:]])

        c = -1.1*np.min(g_gate)
        c = np.maximum(c, 0.1*np.max(g_gate))
        g = g_gate + c
        f = f_gate + c

        g /= np.sum(g*dt)
        f /= np.sum(f*dt)

        # G = np.cumsum(g*dt)
        # F = np.cumsum(f*dt)

        G = integ.cumtrapz(g, t, initial=0)
        F = integ.cumtrapz(f, t, initial=0)

        F_i = ip.interp1d(t, F, fill_value="extrapolate")
        F_1 = ip.interp1d(F, t, fill_value="extrapolate")
        G_1 = ip.interp1d(G, t, fill_value="extrapolate")

        # TT = ip.interp1d(t, G_1(F_i(t)))
        # DD = ip.interp1d(t, np.cumsum(((t[-1] - t) - TT(t[-1] - t))*dt)) # cdf reversa da distancia transportada
        # I = ip.interp1d(t, -t[-1]*DD(t[-1] - t)) # provavelmente tem que multiplicar por 2*dt, mas faz diferença?
        # s_W2[s, :] = I(t) - I(1)# *dt

        dist = ip.interp1d(t, 2*integ.cumtrapz(t-G_1(F_i(t)), initial=0))
        s_W2[s, :] = dist(t) - dist(1)

        W2 += sum(np.abs(F_1(t)-G_1(t))**2)
        # W2 += sum(f*(np.abs(t-G_1(F_i(t)))**2))

    return W2, s_W2


def wasserstein_quad(sim, obs):
    # minha versao
    [n_sens, T] = sim.shape
    t = np.linspace(0,1,T)
    dt = t[1] - t[0]
    W2 = 0
    s_W2 = np.zeros((n_sens, T))
    for s in range(n_sens):
        g_quad = obs[s, :]**2
        f_quad = sim[s, :]**2
        c_quad = np.max(g_quad)*0.05

        g_quad += c_quad
        f_quad += c_quad

        g_quad /= np.sum(g_quad*dt)
        f_quad /= np.sum(f_quad*dt)

        G_quad = np.cumsum(g_quad*dt)
        F_quad = np.cumsum(f_quad*dt)

        F_i_quad = ip.interp1d(t, F_quad, fill_value="extrapolate")
        F_1_quad = ip.interp1d(F_quad, t, fill_value="extrapolate")
        G_1_quad = ip.interp1d(G_quad, t, fill_value="extrapolate")

        # TT = ip.interp1d(t, G_1_quad(F_i_quad(t)))
        # DD = ip.interp1d(t, np.cumsum(((t[-1] - t) - TT(t[-1] - t))*dt)) # cdf reversa da distancia transportada
        # I = ip.interp1d(t, -t[-1]*DD(t[-1] - t)) # provavelmente tem que multiplicar por 2*dt, mas faz diferença?
        # s_W2[s, :] = I(t) # *dt

        dist = ip.interp1d(t, 2*np.cumsum(t-G_1_quad(F_i_quad(t)))*dt)
        s_W2[s, :] = dist(t) - dist(1)

        W2 += sum(np.abs(F_1_quad(t)-G_1_quad(t))**2)
    return W2, s_W2


def wasserstein_experimental(sim, obs):
    # minha versao
    [n_sens, T] = sim.shape
    t = np.linspace(0,1,T)
    dt = t[1] - t[0]
    W2 = 0
    s_W2 = np.zeros((n_sens, T))
    gate_samples = 0
    for s in range(n_sens):

        f_gate = np.concatenate([np.zeros(gate_samples), sim[s, gate_samples:]])
        g_gate = np.concatenate([np.zeros(gate_samples), obs[s, gate_samples:]])

        c = -1.1*np.min(g_gate)
        c = np.maximum(c, 0.1*np.max(g_gate))
        g = g_gate + c
        f = f_gate + c

        g /= np.sum(g*dt)
        f /= np.sum(f*dt)

        G = np.cumsum(g*dt)
        F = np.cumsum(f*dt)

        F_i = ip.interp1d(t, F, fill_value="extrapolate")
        F_1 = ip.interp1d(F, t, fill_value="extrapolate")
        G_1 = ip.interp1d(G, t, fill_value="extrapolate")

        dist = ip.interp1d(t, t*((t-G_1(F_i(t)))**2 - 2*t*(G_1(F_i(t)))))
        s_W2[s, :] = dist(t) - dist(1)

        W2 += sum(np.abs(F_1(t)-G_1(t))**2*F_1(t))
        # W2 += sum(t*np.abs(t-G_1(F_i(t))*f)**2)

    return W2, s_W2


def wasserstein_pm(sim, obs):
    # minha versao
    [n_sens, T] = sim.shape
    t = np.linspace(0,1,T)
    dt = t[1] - t[0]
    W2 = 0
    s_W2 = np.zeros((n_sens, T))
    gate = 0
    for s in range(n_sens):
        g_p = np.maximum(0, obs[s, :])
        f_p = np.maximum(0, sim[s, :])
        g_n = -np.minimum(0, obs[s, :])
        f_n = -np.minimum(0, sim[s, :])

        g_p[:gate] = g_n[:gate] = f_p[:gate] = f_n[:gate] = 0
        c_p = np.max(g_p)*0.01
        c_n = np.max(g_n)*0.01

        g_p += c_p
        f_p += c_p
        g_n += c_n
        f_n += c_n

        mass_gp = np.sum(g_p*dt)
        mass_gn = np.sum(g_n*dt)
        mass_fp = np.sum(f_p*dt)
        mass_fn = np.sum(f_n*dt)

        g_p /= mass_gp
        g_n /= mass_gn
        f_p /= mass_fp
        f_n /= mass_fn

        G_p = np.cumsum(g_p*dt)
        F_p = np.cumsum(f_p*dt)
        G_n = np.cumsum(g_p*dt)
        F_n = np.cumsum(f_p*dt)

        # f_i = ip.interp1d(t, f, fill_value="extrapolate")
        # g_i = ip.interp1d(t, g, fill_value="extrapolate")
        F_ip = ip.interp1d(t, F_p, fill_value="extrapolate")
        G_1p = ip.interp1d(G_p, t, fill_value="extrapolate")
        F_1p = ip.interp1d(F_p, t, fill_value="extrapolate")
        F_in = ip.interp1d(t, F_n, fill_value="extrapolate")
        G_1n = ip.interp1d(G_n, t, fill_value="extrapolate")
        F_1n = ip.interp1d(F_n, t, fill_value="extrapolate")
        # G_i = ip.interp1d(t, G, fill_value="extrapolate")


        TT = ip.interp1d(t, 0.5*(G_1p(F_ip(t)) + G_1n(F_in(t))))
        DD = ip.interp1d(t, np.cumsum(((t[-1] - t) - TT(t[-1] - t))*dt)) # cdf reversa da distancia transportada
        I = ip.interp1d(t, -t[-1]*DD(t[-1] - t)) # provavelmente tem que multiplicar por 2*dt, mas faz diferença?

        # TTp = ip.interp1d(t, G_1p(F_ip(t)))
        # TTn = ip.interp1d(t, G_1n(F_in(t)))
        #
        # DDp = ip.interp1d(t, np.cumsum(((t[-1] - t) - TTp(t[-1] - t))*dt)) # cdf reversa da distancia transportada
        # DDn = ip.interp1d(t, np.cumsum(((t[-1] - t) - TTn(t[-1] - t))*dt)) # cdf reversa da distancia transportada
        # Ip = ip.interp1d(t, -t[-1]*DDp(t[-1] - t)) # provavelmente tem que multiplicar por 2*dt, mas faz diferença?
        # In = ip.interp1d(t, -t[-1]*DDn(t[-1] - t)) # provavelmente tem que multiplicar por 2*dt, mas faz diferença?
        #
        # dist_p = ip.interp1d(t, 2*np.cumsum(t-G_1p(F_ip(t)))*dt)
        # dist_n = ip.interp1d(t, 2*np.cumsum(t-G_1n(F_in(t)))*dt)
        #
        # dist = ip.interp1d(t, dist_p(np.clip(F_ip(t), 0, 1))/2 + dist_n(np.clip(F_in(t), 0, 1)/2))

        # s_W2[s, :] = dist_p(np.clip(F_ip(t), 0, 1))/2 + dist_n(np.clip(F_in(t), 0, 1)/2)

        dist = ip.interp1d(t, np.cumsum(t-G_1p(F_ip(t)))*dt + np.cumsum(t-G_1n(F_in(t)))*dt)

        s_W2[s, :] = I(t)
        s_W2[s, :] = dist(t) - dist(t[-1])
        W2 += sum(np.abs(F_1p(t)-G_1p(t))**2) + sum(np.abs(F_1n(t)-G_1n(t))**2)

        # s_W2[s, :] = mass_fp*dist_p(np.clip(F_ip(t), 0, 1)) + mass_fn*dist_n(np.clip(F_in(t), 0, 1))
        # W2 += sum(np.abs(F_1p(t)-G_1p(t))**2)*mass_fp + sum(np.abs(F_1n(t)-G_1n(t))**2)*mass_fn

        # s_W2[s, :] = Ip(t)*mass_fp + In(t)*mass_fn# *dt
        # W2 += sum(np.abs(F_1p(t)-G_1p(t))**2)*mass_fp + sum(np.abs(F_1n(t)-G_1n(t))**2)*mass_fn
    return W2, s_W2


def wasserstein_yang(ff, gg):
    # precisa debugar
    [n_sens, T] = ff.shape
    t = np.linspace(0,1,T)
    dt = t[1] - t[0]
    W2 = 0
    s_W2 = np.zeros((n_sens, T))

    for s in range(n_sens):
        c = -1.1*np.min(gg[s, :])
        g = gg[s, :] + c
        f = ff[s, :] + c
        g /= np.sum(g*dt) #53e3
        f /= np.sum(f*dt) #53e3

        G = np.cumsum(g*dt)
        F = np.cumsum(f*dt)

        f_i = ip.interp1d(t, f, fill_value="extrapolate")
        g_i = ip.interp1d(t, g, fill_value="extrapolate")
        F_i = ip.interp1d(t, F, fill_value="extrapolate")
        G_i = ip.interp1d(t, G, fill_value="extrapolate")
        G_1 = ip.interp1d(G, t, fill_value="extrapolate")
        F_1 = ip.interp1d(F, t, fill_value="extrapolate")

        # TT = ip.interp1d(t, F_1(G_i(t)))  # invertido em relacao ao artigo
        TT = ip.interp1d(t, G_1(F_i(t)))  # invertido em relacao ao artigo

        DD = ip.interp1d(t, np.cumsum(((t[-1] - t) - TT(t[-1] - t))*dt))
        I = ip.interp1d(t, -t[-1]*DD(t[-1] - t))

        # L = np.tril(np.ones((Nt, Nt)))
        # s_W2 = ((np.diag(-2*f/g_i(G_1(F)))@L)+np.diag(t-G_1(F)))@(t-G_1(F))
        U = np.triu(np.ones((T, T)))
        # s_W2 = (U@np.diag(-2*f/gg(G_1(F)))+np.diag(t-G_1(F)))@(t-G_1(F))
        s_W2[s, :] = (U@np.diag(-2*f_i(t)/g_i(G_1(F_i(t))))+np.diag(t-G_1(F_i(t))))@(t-G_1(F_i(t)))
        # s_W2 = (U@np.diag(-2*f_i(t)/g_i(T(t)))+np.diag(t-T(t)))@(t-T(t))
        # s_W2[s, :] = I(t)*dt
        W2 += sum(np.abs(t-G_1(F))**2*f)

    return W2, s_W2
