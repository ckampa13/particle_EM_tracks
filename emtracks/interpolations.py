import numpy as np
import pandas as pd
from scipy.constants import c
from .tools import InitConds
from .conversions import q_factor

def point_slope(x, xs, ys):
    slope = (ys[1]-ys[0])/(xs[1]-xs[0])
    y_result = slope*(x - xs[0]) + ys[0]
    return y_result

def interp_cole(df, z):
    if z > df.z.max() or z < df.z.min():
        raise ValueError('Invalid input z for interpolation. '+
                         f'Try again with {df.z.min():.3f} <= z <= {df.z.max():.3f}')
    i0 = df[df.z <= z].iloc[-1].name
    if i0 >= len(df)-1:
        i0 = len(df)-2
    interpolants = df.iloc[i0:i0+2]

    t_interps = interpolants.t.values
    x_interps = interpolants.x.values
    y_interps = interpolants.y.values
    z_interps = interpolants.z.values

    t_interp = point_slope(z, z_interps, t_interps)
    x_interp = point_slope(z, z_interps, x_interps)
    y_interp = point_slope(z, z_interps, y_interps)

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
