import math
import numbers
import numpy as np
import pandas as pd
import lmfit as lm
from scipy.spatial.transform import Rotation
import scipy.optimize as optimize

import pypdt
from .conversions import one_gev_c2_to_kg, one_kgm_s_to_mev_c, q_factor
from scipy.constants import c

cbar = c / 1.e9
cmmns = c / 1.e6
m_e = pypdt.get(11).mass*1000.
q_e = -1

# fitting routine
def fit_helix(track_data, B_data, m=m_e, q=q_e, rot=True):
    '''
    track_data is 4 by N np.array, B_data is 3 by N where each row is
    track_data: [t, x, y, z] with units [s, m, m, m]
    B_data: [Bx, By, Bz] with units [T, T, T]
    (to work out of the box with emtracks.particle)
    '''
    track_data = track_data.copy() # to not change track_data information
    track_data[0] = track_data[0]*1e9 # convert to ns
    track_data[1:] = track_data[1:]*1e3 # convert to mm
    # x0, y0 = track_data[1:3,0] # for phi0 estimate
    # translations:
    translate_vec = track_data[:,0].reshape(1, 4)
    track_data = track_data - np.repeat(translate_vec, track_data.shape[1], axis=0).T
    # Rotate so mean B vector is on z-axis
    B_mean = np.mean(B_data, axis=1)
    B = np.linalg.norm(B_mean)
    B_mean_T = np.linalg.norm(B_mean[:2]) # average transverse Bfield (original coords)
    # calculate rotation angles
    rot_theta = np.arctan2(B_mean_T, B_mean[2])
    rot_phi = np.arctan2(B_mean[1], B_mean[0])
    # create rotation function (+inverse) to align mean B with z axis
    rot_func = Rotation.from_euler('zy', np.array([-rot_phi, -rot_theta]))
    rot_inv_func = rot_func.inv()
    # rotate track data
    track_data_rot = track_data.copy()
    if rot:
        track_data_rot[1:] = rot_func.apply(track_data_rot[1:].T).T
    # estimate parameters
    # R, C_x, C_y
    cent, R_guess, Ri_fit, R_residual = reco_circle(track_data_rot[1], track_data_rot[2])
    C_x_guess, C_y_guess = cent
    # Lambda
    dists = (np.sum(np.square(track_data_rot[1:3]), axis=0))**(1/2)
    diffs_negative = np.diff(dists) < 0
    if diffs_negative.sum() == 0:
        endpoint = -1
    else:
        endpoint = np.argwhere(np.diff(dists) < 0).flatten()[0] - 1
    xyz_ends = track_data_rot[1:, [0, endpoint]]
    # print(xyz_ends)
    Lambda_guess = Lambda_est(R_guess, xyz_ends)
    # phi0
    # x0, y0 = track_data_rot[1:3,0] # WRONG-->[0,0,0]
    # x0p = -y0
    # y0p = x0
    # phi0_guess = np.arctan2(y0p, x0p)
    phi0_guess = abs(np.arctan2(C_x_guess, C_y_guess))
    # t0 should be 0 by construction (the translation)
    t0_guess = 0.
    # guess dict
    params_guess = {'R':R_guess, 'Lambda':Lambda_guess, 'C_x':C_x_guess, 'C_y':C_y_guess,
                    'phi0':phi0_guess, 't0':t0_guess}
    mom_guess = LHelix_get_momentum(params_guess['R'], params_guess['Lambda'], m, q, B)
    # construct model
    model = lm.Model(LHelix_P_pos, independent_vars=['t'])
    params = lm.Parameters()
    params.add('R', value=R_guess, min=R_guess-10, max=R_guess+10)
    params.add('Lambda', value=Lambda_guess, min=Lambda_guess-10, max=Lambda_guess+10)
    params.add('C_x', value=C_x_guess, min=C_x_guess-10, max=C_x_guess+10)
    params.add('C_y', value=C_y_guess, min=C_y_guess-10, max=C_y_guess+10)
    params.add('phi0', value=phi0_guess, min=phi0_guess-np.pi/10, max=phi0_guess+np.pi/10)
    # params.add('t0', value=t0_guess, min=t0_guess-1, max=t0_guess+1)#vary=False)
    # params.add('R', value=R_guess, )#min=R_guess-25, max=R_guess+25)
    # params.add('Lambda', value=Lambda_guess,) #min=Lambda_guess-25, max=Lambda_guess+25)
    # params.add('C_x', value=C_x_guess,) #min=C_x_guess-10, max=C_x_guess+10)
    # params.add('C_y', value=C_y_guess,) #min=C_y_guess-10, max=C_y_guess+10)
    # params.add('phi0', value=phi0_guess, min=0., max=2*np.pi)
    params.add('t0', value=t0_guess, vary=False)
    params.add('m', value=m, vary=False)
    params.add('q', value=q, vary=False)
    params.add('B', value=B, vary=False)

    result = model.fit(track_data_rot[1:], t=track_data_rot[0],params=params)
    params_fit = {key:val.value for key,val in result.params.items()}
    mom_fit = LHelix_get_momentum(params_fit['R'], params_fit['Lambda'], m, q, B)

    track_fit_xyz, track_fit_mom, track_fit_mom_vec = LHelix_P(track_data[0], **params_fit)
    # rotate track fit and momentum vec
    if rot:
        track_fit_xyz = rot_inv_func.apply(track_fit_xyz.T).T
        track_fit_mom_vec = rot_inv_func.apply(track_fit_mom_vec.T).T
    track_fit_xyz = track_fit_xyz + np.repeat(translate_vec[:,1:], track_data.shape[1], axis=0).T
    track_fit_xyz = 1e-3*track_fit_xyz
    df_fit = pd.DataFrame({'t':(track_data[0]+translate_vec[:,0])*1e-9, 'x':track_fit_xyz[0],
                           'y':track_fit_xyz[1], 'z':track_fit_xyz[2],
                           'px':track_fit_mom_vec[0], 'py':track_fit_mom_vec[1],
                           'pz':track_fit_mom_vec[2]})
    # return track_data_rot, R_guess, Lambda_guess, C_x_guess, C_y_guess, phi0_guess, t0_guess
    return mom_fit, result, df_fit, params_fit, mom_guess, params_guess
    # return mom_fit, result, params_fit, df_fit

# estimation helper functions
def Lambda_est(R_guess, xyz_ends):
    # R_guess from circle fit
    # xyz_ends 3 by 2 np.array where column 0 start point, column 1 end point
    delta_z = xyz_ends[2,1] - xyz_ends[2,0]
    chord_length = (np.sum(np.square(np.diff(xyz_ends[:2], axis=1))))**(1/2)
    # print(f'chord_length: {chord_length:.3f}, R_guess: {R_guess:.3f}')
    if 2*R_guess < chord_length:
        asin_ = np.pi/2
        # print('arcsin=pi/2')
    else:
        asin_ = np.arcsin(chord_length/(2*R_guess))
    theta_guess = np.arctan2(delta_z, 2*R_guess*asin_)
    return R_guess * np.tan(theta_guess)

# wrapper for position only
def LHelix_P_pos(**kwargs):
    return LHelix_P(**kwargs)[0]

# following helix parameterization used in KinKal (LHeliix, low momentum, arbitrary origin)
def LHelix_P(t, R, Lambda, C_x, C_y, phi0, t0, m, q, B):
    # [ns, mm, mm, mm, mm, rad, ns, MeV/c^2, integer*elementary_charge, Tesla
    dt = t - t0
    # find t format to get pz of correct shape/length
    if isinstance(t, numbers.Number):
        _pz_ones = 1.
    else:
        _pz_ones = np.ones(len(t))
    Q = -q*cbar*B
    mbar = m / Q
    ebar =  (R**2 + Lambda**2 + mbar**2)**(1/2) # name from KinKal

    Omega = math.copysign(1.,mbar) * cmmns / ebar
    x = C_x + R * np.sin(Omega*dt + phi0)
    y = C_y - R * np.cos(Omega*dt + phi0)
    z = Lambda * Omega * dt
    P_x = Q * R * np.cos(Omega*dt + phi0)
    P_y = Q * R * np.sin(Omega*dt + phi0)
    P_z = Q * Lambda * _pz_ones
    P_t = abs(Q) * ebar
    mom = P_t
    pos = np.array([x,y,z])
    mom_vec = np.array([P_x, P_y, P_z])
    return pos, mom, mom_vec

def LHelix_get_momentum(R, Lambda, m, q, B):
    Q = -q*cbar*B
    mbar = m / Q
    ebar =  (R**2 + Lambda**2 + mbar**2)**(1/2) # name from KinKal
    return abs(Q) * ebar

## CIRCLE FIT FUNCS
# circle fit
def calc_R(xc, yc, x, y):
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def circ_alg_dist(center, x, y):
    Ri = calc_R(*center, x, y)
    return Ri - Ri.mean()

def reco_circle(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_est = x_m, y_m
    center_fit, ier = optimize.leastsq(circ_alg_dist, center_est, args=(x, y))
    Ri_fit = calc_R(*center_fit, x, y)
    R_fit = np.mean(Ri_fit)
    R_residual = np.sum((Ri_fit - R_fit)**2)
    return center_fit, R_fit, Ri_fit, R_residual

# full reco
def reco_arc(df, B_func):
    x, y, z, t, v, pT, pz, p, E = df[['x','y','z','t', 'v', 'pT', 'pz', 'p','E']].values.T
    # 1. reco circle
    center, R, Ri, res = reco_circle(x, y)
    # 2. Calculate theta
    thetai = np.arctan2(y-center[1], x-center[0])
    if not (thetai[0] <= 0. and thetai[-1] > 0.):
        thetai = (thetai + 2 * np.pi) % (2 * np.pi)
    arcangle = thetai[-1] - thetai[0]
    # 3. Calculate arc length
    arclength = R * arcangle
    # 4. Calculate z length
    G = np.array([np.ones_like(t), t]).T
    GtGinv = np.linalg.inv(G.T @ G)
    m = GtGinv @ G.T @ z
    # m[0] intercept, m[1] slope (aka speed in z direction, m / s)
    zlength = m[1] * (t[-1] - t[0])
    vz = m[1]
    # calculate vT
    vT = arclength / (t[-1] - t[0])
    # calculate v
    v = (vT**2 + vz**2)**(1/2)
    # calculate beta
    beta = v / c
    # 5. p, v, etc.
    tantheta = arclength / zlength
    gamma = (1 - beta**2)**(-1/2)
    Bxs, Bys, Bzs = np.array([B_func([xi,yi,zi]) for xi,yi,zi in zip(x,y,z)]).T
    Bs = (Bxs**2 + Bys**2 + Bzs**2)**(1/2)
    BTs = (Bxs**2 + Bys**2)**(1/2)
    # NEED TO FIND BEST B
    B = Bzs.mean()#Bs.mean() - BTs.mean()#Bzs.min()#Bzs.mean()
    pT = q_factor * B * R * 1000.
    pz = pT / tantheta
    p = (pT**2 + pz**2)**(1/2)
    mass = p * c / (gamma * v )
    E = (p**2 + mass**2)**(1/2)
    # charge
    charge_sign = - np.sign(m[1]) * np.sign(np.arctan2(arclength, zlength))
    return charge_sign, mass, pT, pz, p, E, v

