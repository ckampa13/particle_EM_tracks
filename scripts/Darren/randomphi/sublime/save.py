import numpy as np
from scipy.constants import c, elementary_charge
import pandas as pd
import pickle as pkl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = [24,16] # bigger figures
from matplotlib import style
style.use('fivethirtyeight')

from emtracks.particle import trajectory_solver # main solver object
from emtracks.conversions import one_gev_c2_to_kg # conversion for q factor (transverse momentum estimate)
from emtracks.tools import *#InitConds # initial conditions namedtuple
from emtracks.mapinterp import get_df_interp_func  # factory function for creating Mu2e DS interpolation function
from emtracks.Bdist import get_B_df_distorted
import matplotlib.animation as animation

testdir = "/home/darren/Desktop/plots/"
datadir = "/home/shared_data/"
plotdir = datadir+"plots/randomphi/"
mapdir = datadir+"Bmaps/"
date = "/6-20/"

#DO NOT CHANGE ON THIS DOCUMENT
start_point = 3
end_point = 14
initial_B = 50 #(rougly 1% distortion at z = 3.0, 0% at z = 14)
final_B = 0

m = (final_B - initial_B) / (end_point - start_point)
n = 50
step = (end_point - start_point) / n
t = np.arange(start_point, end_point, step)
x = plt.plot(t, ((t - start_point)*m) + initial_B)
plt.title("Distortion")
plt.xlabel("Z (meters)")
plt.ylabel("B (gauss)")

#MU2E FIELD
df_Mu2e = pd.read_pickle(mapdir+"Mu2e_DSMap_V13.p")
B_Mu2e_func = get_df_interp_func(mapdir+"Mu2e_DSMap_V13.p", gauss=False)

#MU2E FIELD + DIS
df_Mu2e_dis = get_B_df_distorted(df_Mu2e, v="0", Bz0 = initial_B, Bzf = 0, z0 = start_point, zf = end_point)
B_Mu2e_dis = get_df_interp_func(df=df_Mu2e_dis, gauss=False)


#input N, return N random values between 0 and 2pi
def get_random_phi(N):
    phis = np.random.uniform(0, 2*math.pi, N)
    return phis

#input N, return N equally spaced values between 0 and 2pi
def get_uniform_phi(N):
    phis = np.linspace(0, 2*math.pi, N)
    return phis

#input list of phis, number of steps for integrator, initial position / return dataframe trajectory
def run_solver(phi, N_calc, field, xnaught, ynaught, znaught, name):
    ic_Mu2e = InitConds(t0=0., tf=4e-8, N_t=N_calc,
                    x0=xnaught, y0=ynaught, z0=znaught,
                    p0=104.96, theta0=np.pi/3, phi0=phi)
    e_solver = trajectory_solver(ic_Mu2e, B_func=field, bounds=bounds_Mu2e)
    sol = e_solver.solve_trajectory(verbose = False, atol=1e-10, rtol=1e-10) # high tolerance so it runs quickly for testing
    e_solver.dataframe['r'] = ((e_solver.dataframe['x'])**2 + (e_solver.dataframe['y'])**2)**(1/2)
    e_solver.to_pickle(datadir+f'50_Gauss_z_3-14/{phi}_{name}.pkl')
    
    return e_solver.dataframe