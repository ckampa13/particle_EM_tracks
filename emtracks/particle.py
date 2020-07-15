import time
import math
import copy
import numpy as np
import pandas as pd
import dill as pkl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import c, elementary_charge
from scipy.integrate import solve_ivp
import scipy.optimize as optimize

import pypdt
from .helix import *
from .conversions import one_gev_c2_to_kg, one_kgm_s_to_mev_c, q_factor
from .plotting import config_plots
config_plots()


class trajectory_solver(object):
    def __init__(self, init_conds=None, bounds=None, particle_id=11, B_func=lambda p_vec: np.array([0.,0.,1.]), E_func = lambda p_vec: np.array([0., 0., 0.]), zevents=None):
        '''
        Default particle_id is for electron

        B_func should take in 3D numpy array of position, but defaults to a 1 Tesla field in the z direction.
        '''
        self.init_conds = init_conds
        self.bounds = bounds
        # dummy termination (no termination) if no bounds
        if bounds is None:
            self.terminate = lambda t, x: 1
        else:
            self.terminate = get_terminate(bounds)
        if particle_id < 0:
            sign = -1
        else:
            sign = +1
        particle_id = abs(particle_id)
        self.particle = pypdt.get(particle_id)
        self.particle.mass_kg = self.particle.mass * one_gev_c2_to_kg
        self.particle.charge_true = self.particle.charge*sign
        self.B_func = B_func
        self.E_func = E_func
        self.zevents = zevents

    @classmethod
    def from_pickle(cls, filename):
        return pkl.load(open(filename, "rb"))

    def to_pickle(self, filename):
        # delete B_func to avoid bloat
        # should include B_func meta data somewhere. FIX ME
        B_ = copy.deepcopy(self.B_func)
        del(self.B_func)
        # use dill package as "pkl" for picking lambda fcns
        pkl.dump(self, file=open(filename, 'wb'))
        # reset the B_func
        self.B_func = B_

    def z_event_func(self, t, y):
        # can define inside class since we don't need to set terminal=True
        z = y[2]
        if self.zevents is None:
            return 1
        # conds = [z == z_ for z_ in self.zevents]
        conds = np.isclose(z, self.zevents, atol=1e-2, rtol=1e-2)
        return int(not any(conds))

    def gamma(self, v_vec):
        '''
        Calculate gamma factor given a velocity (m / s),

        Args: v_vec is a 3 element np.array
        '''
        beta = v_vec / c
        return 1 / math.sqrt(1 - np.dot(beta, beta))

    def lorentz_accel(self, B_vec, E_vec, mom_vec, gamma):
        '''
        Calculate Lorentz acceleration
        '''
        a = self.particle.charge_true * elementary_charge * (E_vec*one_kgm_s_to_mev_c + 1. / (gamma*self.particle.mass_kg) * np.cross(mom_vec, B_vec))
        return a

    def lorentz_update(self, t, y):
        '''
        Function to have solve_ivp solve
        '''
        pos_vec = np.array(y[:3]) # meters
        mom_vec = np.array(y[3:]) # MeV / c
        v_vec = c * mom_vec / np.sqrt(np.dot(mom_vec, mom_vec) + (self.particle.mass*1000.)**2)
        B_vec = self.B_func(pos_vec) # B_vec based on position
        E_vec = self.E_func(pos_vec) # E_vec based on position
        gamma = self.gamma(v_vec) # gamma based on velocity
        f = [None] * 6 # [None, None, None, None, None, None]
        f[:3] = v_vec # y[3:] # position is updated by current velocity
        f[3:] = self.lorentz_accel(B_vec, E_vec, mom_vec, gamma)
        return f

    def solve_trajectory(self, verbose=False, method='DOP853', atol=1e-10, rtol=1e-10, dense=False):
        t_span = (self.init_conds.t0, self.init_conds.tf)
        t_eval = np.linspace(self.init_conds.t0, self.init_conds.tf, self.init_conds.N_t)
        p0 = self.init_conds.p0
        px0 = p0 * np.sin(self.init_conds.theta0) * np.cos(self.init_conds.phi0)
        py0 = p0 * np.sin(self.init_conds.theta0) * np.sin(self.init_conds.phi0)
        pz0 = p0 * np.cos(self.init_conds.theta0)
        y_init = [self.init_conds.x0, self.init_conds.y0, self.init_conds.z0, px0, py0, pz0]
        if self.zevents is None:
            events_list = [self.terminate]
        else:
            events_list = [self.terminate, self.z_event_func]
        if verbose:
            print(f"y_init: {y_init}")
        start_time = time.time()
        sol = solve_ivp(self.lorentz_update, t_span=t_span, y0 = y_init,\
                method=method, atol=atol, rtol=rtol, t_eval=t_eval,\
                events=events_list, dense_output=dense)
        t = sol.t
        t_e = sol.t_events
        x, y, z, px, py, pz = sol.y
        # x_e, y_e, z_e, px_e, py_e, pz_e = sol.y_events
        self.dataframe = pd.DataFrame({"t":t,"x":x,"y":y,"z":z,"px":px,"py":py,"pz":pz})
        # self.events_df = pd.DataFrame({"t":t_e,"x":x_e,"y":y_e,"z":z_e,"px":px_e,"py":py_e,"pz":pz_e})
        # calculate extra interesting variables
        # for df in [self.dataframe, self.events_df]:
        for df in [self.dataframe,]:
            df.eval("pT = (px**2 + py**2)**(1/2)", inplace=True)
            df.eval("p = (px**2 + py**2 + pz**2)**(1/2)", inplace=True)
            df.eval("E = (p**2 + (@self.particle.mass*1000.)**2)**(1/2)", inplace=True)
            df.eval("beta = p / E", inplace=True)
            df.eval("v = beta * @c", inplace=True)
            df.eval("vx = @c * px / E", inplace=True)
            df.eval("vy = @c * py / E", inplace=True)
            df.eval("vz = @c * pz / E", inplace=True)
            df.loc[:, "theta"] = np.arctan2(df.pT, df.pz)
            df.loc[:, "phi"] = np.arctan2(df.py, df.px)
        end_time = time.time()
        if verbose:
            print("Trajectory calculation complete!")
            print(f"Runtime: {(end_time - start_time):.4f} s")
        # return solution object for further use by user
        return sol

    def analyze_trajectory_LHelix(self, query="z >= 8.41 & z <= 11.66", step=100, stride=1):
        if query is None:
            df = self.dataframe.copy()
        else:
            df = self.dataframe.query(query)
        N_steps = int(len(df) // step)
        # pTs, pzs, ps, Es = [], [], [], []
        # pTs_nr, pzs_nr, ps_nr, Es_nr = [], [], [], []
        ps, ps_nr = [], []
        ps_guess, ps_guess_nr = [], []
        consistents = []
        df_fits = []
        df_fits_nr = []
        for i in range(N_steps):
            start = i * step
            stop = start + step
            df_ = df[start:stop:stride]
            track_data = df_[['t','x','y','z']].copy().values.T
            B_data = np.array([self.B_func(coord) for coord in track_data[1:].T]).T
            mom_fit, result, df_fit, params_fit, mom_guess, params_guess = fit_helix(track_data, B_data)
            mom_fit_nr, result_nr, df_fit_nr, params_fit_nr, mom_guess_nr, params_guess_nr = fit_helix(track_data, B_data, rot=False)
            if abs(mom_fit - mom_guess) / mom_guess < 1e-3:
                consistents.append(True)
            else:
                consistents.append(False)
            df_fits.append(df_fit)
            df_fits_nr.append(df_fit_nr)
            ps.append(mom_fit)
            ps_nr.append(mom_fit_nr)
            ps_guess.append(mom_guess)
            ps_guess_nr.append(mom_guess_nr)

        if len(df_fits) > 1:
            self.helix_reco_df = pd.concat(df_fits, ignore_index=True)
            self.no_rotation_helix_reco_df = pd.concat(df_fits_nr, ignore_index=True)
        else:
            self.helix_reco_df = df_fits
            self.no_rotation_helix_reco_df = df_fits_nr
        self.helix_reco_moms_df = pd.DataFrame({'p':ps, 'p_no_rotation':ps_nr, 'consistent':consistents})
        self.guess_helix_reco_moms_df = pd.DataFrame({'p':ps_guess, 'p_no_rotation':ps_guess_nr})
        if self.helix_reco_moms_df.consistent.sum() > 0:
            self.mom_LHelix = self.helix_reco_moms_df.query("consistent==True")['p'].mean()
            self.LHelix_success = True
        else:
            self.mom_LHelix = self.guess_helix_reco_moms_df['p'].mean()
            self.LHelix_success = False

    def analyze_trajectory(self, query=None, B=None, step=100, stride=10):
	# step = # rows to use for each arc segment
	# stride = points between each included reco point
	# N points in track reco = step // stride
        if B is None:
            B = self.B_func
        if query is None:
            df = self.dataframe.copy()
        else:
            df = self.dataframe.query(query)
        # print(df.head())

        N_steps = int(len(df) // step)
        charge_signs, masses, pTs, pzs, ps, Es, vs = [], [], [], [], [], [], []
        # ts_, ps_,
        # charge_signs, masses, pTs, pzs, ps, Es = [], [], [], [], [], []
        for i in range(N_steps):
            start = i * step
            stop = start + step
            df_ = df[start:stop:stride]
            ch, m, pT, pz, p, E, v = reco_arc(df_, B)
       	    charge_signs.append(ch)
            masses.append(m)
            pTs.append(pT)
            pzs.append(pz)
            ps.append(p)
            Es.append(E)
            vs.append(v)

        charge_signs = np.array(charge_signs)
        masses = np.array(masses)
        pTs = np.array(pTs)
        pzs = np.array(pzs)
        ps = np.array(ps)
        Es = np.array(Es)
        vs = np.array(vs)

        self.df_reco = pd.DataFrame({'p':ps, 'E':Es, 'm':masses, 'charge':charge_signs, 'pT':pTs, 'pz': pzs, 'v': vs})


    def plot3d(self, fig=None, ax=None, cmap="viridis"):
        if fig is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.dataframe.x, self.dataframe.y, self.dataframe.z, 'k-', alpha=0.2, zorder=99)
        p = ax.scatter(self.dataframe.x, self.dataframe.y, self.dataframe.z, c=self.dataframe.t, cmap=cmap, s=2, alpha=1., zorder=101)
        cb = fig.colorbar(p)
        cb.set_label('t [s]', rotation=0.)
        ax.set_xlabel('\nX [m]', linespacing=3.0)
        ax.set_ylabel('\nY [m]', linespacing=3.0)
        ax.set_zlabel('\nZ [m]', linespacing=3.0)
        fig.tight_layout()
        def on_draw(event):
            reorder_camera_distance(ax, p)
        fig.canvas.mpl_connect('draw_event', on_draw)
        return fig, ax

    def plot2d(self):
        fig, axs = plt.subplots(3, 3)
        axs[0, 0].plot(self.dataframe.t, self.dataframe.x, 'bo-', markersize=2)
        axs[0, 0].set(xlabel="t [s]", ylabel="x [m]")
        axs[0, 1].plot(self.dataframe.t, self.dataframe.vx, 'bo-', markersize=2)
        axs[0, 1].set(xlabel="t [s]", ylabel="vx [m/s]")
        axs[0, 2].plot(self.dataframe.t, self.dataframe.px, 'bo-', markersize=2)
        axs[0, 2].set(xlabel="t [s]", ylabel="px [MeV/c]")
        axs[1, 0].plot(self.dataframe.t, self.dataframe.y, 'go-', markersize=2)
        axs[1, 0].set(xlabel="t [s]", ylabel="y [m]")
        axs[1, 1].plot(self.dataframe.t, self.dataframe.vy, 'go-', markersize=2)
        axs[1, 1].set(xlabel="t [s]", ylabel="vy [m/s]")
        axs[1, 2].plot(self.dataframe.t, self.dataframe.py, 'bo-', markersize=2)
        axs[1, 2].set(xlabel="t [s]", ylabel="py [MeV/c]")
        axs[2, 0].plot(self.dataframe.t, self.dataframe.z, 'ro-', markersize=2)
        axs[2, 0].set(xlabel="t [s]", ylabel="z [m]")
        axs[2, 1].plot(self.dataframe.t, self.dataframe.vz, 'ro-', markersize=2)
        axs[2, 1].set(xlabel="t [s]", ylabel="vz [m/s]")
        axs[2, 2].plot(self.dataframe.t, self.dataframe.pz, 'bo-', markersize=2)
        axs[2, 2].set(xlabel="t [s]", ylabel="pz [MeV/c]")
        fig.tight_layout()
        return fig, axs

# termination
def get_terminate(bounds):
    def terminate(t, y):
        pos = y[:3]
        conds = [pos[0]<bounds.xmin, pos[0]>bounds.xmax,
                 pos[1]<bounds.ymin, pos[1]>bounds.ymax,
                 pos[2]<bounds.zmin, pos[2]>bounds.zmax]
        return int(not any(conds))
    terminate.terminal = True
    return terminate

# PLOTTING ORDER FIX
def get_camera_position(ax):
    """returns the camera position for 3D axes in cartesian coordinates"""
    r = np.square(ax.xy_viewLim.max).sum()
    theta, phi = np.radians((90 - ax.elev, ax.azim))
    return np.array(sph2cart(r, theta, phi), ndmin=2).T

def sph2cart(r, theta, phi):
    """spherical to cartesian transformation."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def reorder_camera_distance(ax, patches):
    """
    Sort the patches (via their offsets) by decreasing distance from camera position
    so that the furthest gets drawn first.
    """
    # camera position in xyz
    camera = get_camera_position(ax)
    # distance of patches from camera
    d = np.square(np.subtract(camera, patches._offsets3d)).sum(0)
    o = d.argsort()[::-1]

    patches._offsets3d = tuple(np.array(patches._offsets3d)[:, o])
    patches._facecolor3d = patches._facecolor3d[o]
    patches._edgecolor3d = patches._edgecolor3d[o]
    # todo: similar for linestyles, linewidths, etc....

# def on_draw(event):
#     reorder_camera_distance()

