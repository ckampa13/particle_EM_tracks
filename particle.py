import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import c, elementary_charge
from scipy.integrate import solve_ivp
import pypdt
from conversions import one_gev_c2_to_kg
from plotting import config_plots
config_plots()


class trajectory_solver(object):
    def __init__(self, init_conds=None, particle_id=11, B_func=lambda p_vec: np.array([0.,0.,1.])):
        '''
        Default particle_id is for electron

        B_func should take in 3D numpy array of position, but defaults to a 1 Tesla field in the z direction.
        '''
        self.init_conds = init_conds
        self.particle = pypdt.get(particle_id)
        self.particle.mass_kg = self.particle.mass * one_gev_c2_to_kg
        self.B_func = B_func

    def set_init_conds(self, init_conds):
        self.init_conds = init_conds

    def gamma(self, v):
        '''
        Calculate gamma factor given a velocity (m / s),

        Args: v is a 3 element np.array
        '''
        beta = v / c
        return 1 / math.sqrt(1 - np.dot(beta, beta))

    def lorentz_accel(self, B_vec, v_vec, gamma):
        '''
        Calculate Lorentz acceleration
        '''
        a = self.particle.charge * elementary_charge / (gamma*self.particle.mass_kg) * np.cross(v_vec, B_vec)
        return a

    def lorentz_update(self, t, y):
        '''
        Function to have solve_ivp solve
        '''
        p_vec = np.array(y[:3])
        v_vec = np.array(y[3:])
        B_vec = self.B_func(p_vec) # B_vec based on position
        gamma = self.gamma(v_vec) # gamma based on velocity
        f = [None] * 6
        f[:3] = y[3:] # position is updated by current velocity
        f[3:] = self.lorentz_accel(B_vec, v_vec, gamma)
        return f

    def solve_trajectory(self, method='DOP853', atol=1e-10, rtol=1e-10):
        t_span = (self.init_conds.t0, self.init_conds.tf)
        t_eval = np.linspace(self.init_conds.t0, self.init_conds.tf, self.init_conds.N_t)
        v0 = c * self.init_conds.p0 / (self.init_conds.p0**2 + (self.particle.mass*1000.)**2)**(1/2)
        vz0 = v0 * np.cos(self.init_conds.theta0)
        vx0 = v0 * np.sin(self.init_conds.theta0) * np.cos(self.init_conds.phi0)
        vy0 = v0 * np.sin(self.init_conds.theta0) * np.sin(self.init_conds.phi0)
        y_init = [self.init_conds.x0, self.init_conds.y0, self.init_conds.z0, vx0, vy0, vz0]
        print(f"y_init: {y_init}")
        sol = solve_ivp(self.lorentz_update, t_span=t_span, y0 = y_init,\
                method=method, atol=atol, rtol=rtol, t_eval=t_eval)
        self.t = sol.t
        self.x, self.y, self.z, self.vx, self.vy, self.vz = [pd.Series(yi) for yi in sol.y]
        print("Trajectory calculation complete!")
        return sol

    def plot3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.x, self.y, self.z, 'bo-', markersize=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.set_title('Motion of Particle in Constant B field Facing Z-Direction')
        return fig, ax

    def plot2d(self):
        fig, axs = plt.subplots(3, 2)
        axs[0, 0].plot(self.t, self.x, 'bo-', markersize=2)
        axs[0, 0].set(xlabel="t [s]", ylabel="x [m]")
        axs[0, 1].plot(self.t, self.vx, 'bo-', markersize=2)
        axs[0, 1].set(xlabel="t [s]", ylabel="vx [m/s]")
        axs[1, 0].plot(self.t, self.y, 'go-', markersize=2)
        axs[1, 0].set(xlabel="t [s]", ylabel="y [m]")
        axs[1, 1].plot(self.t, self.vy, 'go-', markersize=2)
        axs[1, 1].set(xlabel="t [s]", ylabel="vy [m/s]")
        axs[2, 0].plot(self.t, self.z, 'ro-', markersize=2)
        axs[2, 0].set(xlabel="t [s]", ylabel="z [m]")
        axs[2, 1].plot(self.t, self.vz, 'ro-', markersize=2)
        axs[2, 1].set(xlabel="t [s]", ylabel="vz [m/s]")

        return fig, axs
