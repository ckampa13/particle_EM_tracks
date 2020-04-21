import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import c, elementary_charge
from scipy.integrate import solve_ivp

import pypdt
from conversions import one_gev_c2_to_kg, one_kgm_s_to_mev_c
from plotting import config_plots
config_plots()


class trajectory_solver(object):
    def __init__(self, init_conds=None, particle_id=11, B_func=lambda p_vec: np.array([0.,0.,1.]), E_func = lambda p_vec: np.array([0., 0., 0.])):
        '''
        Default particle_id is for electron

        B_func should take in 3D numpy array of position, but defaults to a 1 Tesla field in the z direction.
        '''
        self.init_conds = init_conds
        self.particle = pypdt.get(particle_id)
        self.particle.mass_kg = self.particle.mass * one_gev_c2_to_kg
        self.B_func = B_func
        self.E_func = E_func

    # def set_init_conds(self, init_conds):
    #     self.init_conds = init_conds

    def gamma(self, v_vec):
        '''
        Calculate gamma factor given a velocity (m / s),

        Args: v_vec is a 3 element np.array
        '''
        beta = v_vec / c
        # betax, betay, betaz
        # beta \dot beta = betax^2 + betay^2 + betaz^2
        #np.sum(beta**2)
        # gamma = 1 / sqrt(1- v^2 / c^2)
        return 1 / math.sqrt(1 - np.dot(beta, beta))

    def lorentz_accel(self, B_vec, E_vec, mom_vec, gamma):
        '''
        Calculate Lorentz acceleration
        '''
        # a = kg m /s
        a = self.particle.charge * elementary_charge * (E_vec + 1. / (gamma*self.particle.mass_kg) * np.cross(mom_vec / one_kgm_s_to_mev_c, B_vec))
        # a --> MeV / c
        return a * one_kgm_s_to_mev_c

    def lorentz_update(self, t, y):
        '''
        Function to have solve_ivp solve
        '''
        pos_vec = np.array(y[:3]) # meters
        mom_vec = np.array(y[3:]) # MeV / c
        # beta = p / E
        # E = (|p|**2 + m**2)**(1/2) # c = 1
        # momentum: MeV, m: MeV
        # E: MeV
        # p / E = unitless (if c=1)
        # p / E = 1 / c (if c != 1)
        v_vec = c * mom_vec / np.sqrt(np.dot(mom_vec, mom_vec) + (self.particle.mass*1000.)**2)
        # E = (p**2 * c**2 + m**2 * c**4)**(1/2)
        # v_vec = c * mom_vec / np.sqrt(mom_vec @ mom_vec + (self.particle.mass*1000)**2)
        # v_vec = np.array(y[3:])
        B_vec = self.B_func(pos_vec) # B_vec based on position
        E_vec = self.E_func(pos_vec) # E_vec based on position
        gamma = self.gamma(v_vec) # gamma based on velocity
        f = [None] * 6 # [None, None, None, None, None, None]

        f[:3] = v_vec # y[3:] # position is updated by current velocity

        f[3:] = self.lorentz_accel(B_vec, E_vec, mom_vec, gamma)
        return f

    def solve_trajectory(self, method='DOP853', atol=1e-10, rtol=1e-10):
    # def solve_trajectory(self, atol=1e-10, rtol=1e-10):
        t_span = (self.init_conds.t0, self.init_conds.tf)
        t_eval = np.linspace(self.init_conds.t0, self.init_conds.tf, self.init_conds.N_t)
        p0 = self.init_conds.p0
        px0 = p0 * np.sin(self.init_conds.theta0) * np.cos(self.init_conds.phi0)
        py0 = p0 * np.sin(self.init_conds.theta0) * np.sin(self.init_conds.phi0)
        pz0 = p0 * np.cos(self.init_conds.theta0)
        y_init = [self.init_conds.x0, self.init_conds.y0, self.init_conds.z0, px0, py0, pz0]
        print(f"y_init: {y_init}")

        sol = solve_ivp(self.lorentz_update, t_span=t_span, y0 = y_init,\
                method=method, atol=atol, rtol=rtol, t_eval=t_eval)
        self.t = pd.Series(sol.t)
        self.x, self.y, self.z, self.px, self.py, self.pz = [pd.Series(yi) for yi in sol.y]
        self.pT = np.sqrt(self.px**2 + self.py**2)
        self.p = np.sqrt(self.px**2 + self.py**2 + self.pz**2)
        self.E = np.sqrt(self.p**2 + (self.particle.mass*1000.)**2)
        self.beta = self.p / self.E
        self.v = self.beta * c
        self.vx = c * self.px / self.E
        self.vy = c * self.py / self.E
        self.vz = c * self.pz / self.E
        print("Trajectory calculation complete!")
        return sol

    def plot3d(self):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.x, self.y, self.z, 'k-', alpha=0.2, zorder=99)
        p = ax.scatter(self.x, self.y, self.z, c=self.t, cmap="viridis", s=2, alpha=1., zorder=101)
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
        axs[0, 0].plot(self.t, self.x, 'bo-', markersize=2)
        axs[0, 0].set(xlabel="t [s]", ylabel="x [m]")
        axs[0, 1].plot(self.t, self.vx, 'bo-', markersize=2)
        axs[0, 1].set(xlabel="t [s]", ylabel="vx [m/s]")
        axs[0, 2].plot(self.t, self.px, 'bo-', markersize=2)
        axs[0, 2].set(xlabel="t [s]", ylabel="px [MeV/c]")
        axs[1, 0].plot(self.t, self.y, 'go-', markersize=2)
        axs[1, 0].set(xlabel="t [s]", ylabel="y [m]")
        axs[1, 1].plot(self.t, self.vy, 'go-', markersize=2)
        axs[1, 1].set(xlabel="t [s]", ylabel="vy [m/s]")
        axs[1, 2].plot(self.t, self.py, 'bo-', markersize=2)
        axs[1, 2].set(xlabel="t [s]", ylabel="py [MeV/c]")
        axs[2, 0].plot(self.t, self.z, 'ro-', markersize=2)
        axs[2, 0].set(xlabel="t [s]", ylabel="z [m]")
        axs[2, 1].plot(self.t, self.vz, 'ro-', markersize=2)
        axs[2, 1].set(xlabel="t [s]", ylabel="vz [m/s]")
        axs[2, 2].plot(self.t, self.pz, 'bo-', markersize=2)
        axs[2, 2].set(xlabel="t [s]", ylabel="pz [MeV/c]")
        return fig, axs


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

