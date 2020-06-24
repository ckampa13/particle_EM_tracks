import numpy as np
import pandas as pd
from scipy.constants import c
from .tools import InitConds
from .conversions import q_factor

def interp_cole(df, z):
    # # average z distance between points
    # delta = (df.z.max() - df.z.min()) / len(df)
    # # # delta = 10/4001   #approximate z range divided by number of points
    # mask = (df.z < z + delta) & (df.z > z - delta)

    # while (len(df.z[mask]) != 2):
    #     if len(df.z[mask]) > 2:
    #         delta = delta / 1.5
    #     else:
    #         delta = delta * 2
    #     mask = (df.z < z + delta) & (df.z > z - delta)

    # interpolants = df[mask]

    try:
        i0 = df[df.z < z].iloc[-1].name
    except:
        i0 = df[df.z >= z].iloc[0].name
    if i0 >= len(df)-1:
        i0 = len(df)-2
    interpolants = df.iloc[i0:i0+2]

    t_interps = interpolants.t.values
    x_interps = interpolants.x.values
    y_interps = interpolants.y.values
    z_interps = interpolants.z.values

    def t(z):
        slope = (t_interps[1]-t_interps[0])/(z_interps[1]-z_interps[0])
        t_res = slope*(z - z_interps[0]) + t_interps[0]
        return t_res
    def x(z):
        slope = (x_interps[1]-x_interps[0])/(z_interps[1]-z_interps[0])
        x_res = slope*(z - z_interps[0]) + x_interps[0]
        return x_res
    def y(z):
        slope = (y_interps[1]-y_interps[0])/(z_interps[1]-z_interps[0])
        y_res = slope*(z - z_interps[0]) + y_interps[0]
        return y_res

    t_interp = t(z)
    x_interp = x(z)
    y_interp = y(z)

    return t_interp, x_interp, y_interp


def get_ic_analytical(R, B, theta0, tf = 5e-8, N_t=5001):
    x0 = R
    y0 = 0.
    z0 = 0.
    t0 = 0.
    pT = q_factor * R * B
    pz = pT / np.tan(theta0)
    p0 = (pz**2 + pT**2)**(1/2)
    phi0 = np.pi/2
    ic = InitConds(t0=t0, tf=tf, N_t=N_t, x0=x0, y0=y0, z0=z0, p0=p0*1000., theta0=theta0, phi0=phi0)
    return ic

def interp_analytical(ic, m, z):
    # m == MeV/c
    R = ic.x0
    E = (ic.p0**2 + m**2)**(1/2)
    beta = ic.p0 / E
    v = beta * c
    vT = v * np.sin(ic.theta0)
    vz = v * np.cos(ic.theta0)
    period = 2*np.pi * R / vT
    const = 2*np.pi / period
    t_true = (z - ic.z0) / vz
    x_true = R * np.cos(const * t_true)
    y_true = R * np.sin(const * t_true)
    return t_true, x_true, y_true
