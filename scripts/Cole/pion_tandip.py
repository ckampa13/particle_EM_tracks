import sys
import time
# import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

from emtracks.particle import trajectory_solver
from emtracks.tools import InitConds, Bounds, ic_Mu2e, bounds_Mu2e
from emtracks.mapinterp import get_df_interp_func
from emtracks.plotting import config_plots

config_plots()

# default Bfield (Mau13)
ddir = "/home/ckampa/data/"
fBnom = ddir+"Bmaps/Mu2e_DSMap_V13.p"

def generate_IC_df(N_gen=100, zupstream=4.235, halflength=1.75e-3, R=0.1727, momentum=105.):
    # generate uniform positions in a cylinder (pion degrader)
    zs = np.random.uniform(zupstream-halflength, zupstream+halflength, size=N_gen)
    phis = np.random.uniform(0, 2*np.pi, size=N_gen)
    Rs = R*(np.random.rand(N_gen))**(1/2)
    xs = Rs * np.cos(phis)
    ys = Rs * np.sin(phis)
    # generate uniform momentum vector
    p_costhetas = np.random.uniform(-1, 1, size=N_gen)
    p_thetas = np.arccos(p_costhetas)
    p_phis = np.random.uniform(0, 2*np.pi, size=N_gen)
    # populate pandas df and add |p| column
    df = pd.DataFrame({'x':xs, 'y':ys, 'z':zs, 'theta':p_thetas, 'phi':p_phis})
    df['p'] = momentum

    return df

def run_trajectory(ic_df_row, B_func, pid=-11, z_tand=10.175-1.6351):
    ic_ = InitConds(t0=0, tf=7e-7, N_t=70001,
                    x0=ic_df_row.x, y0=ic_df_row.y, z0=ic_df_row.z,
                    p0=ic_df_row.p, theta0=ic_df_row.theta, phi0=ic_df_row.phi)
    e = trajectory_solver(init_conds=ic_, bounds=bounds_Mu2e, particle_id=pid, B_func=B_func)
    e.solve_trajectory(atol=1e-8, rtol=1e-8)
    df_ = e.dataframe[e.dataframe.z >= z_tand]
    if len(df_) >= 1:
        tand_tracker = (df_.pz / df_.pT).iloc[0]
        df_.eval('r = (x**2 + y**2)**(1/2)', inplace=True)
        Rmax = df_.r.max()
    else:
        tand_tracker = -1000
        Rmax = -1000
    tand_init = np.tan(np.pi/2 - ic_df_row.theta)
    return tand_init, tand_tracker, Rmax

def map_run(ic_df, B_file=fBnom, name='Mau13'):
    # load Bmap
    B_ = get_df_interp_func(filename=B_file, gauss=False)
    print("Start of emtracks Sim in Pion Degrader (tan(dip))")
    print("-----------------------------------------")
    # print(f"Directory: {outdir},\nFilename: {name},\nNumber: {N}")
    print(f"Using BField File: {B_file}")
    start = time.time()
    num_cpu = multiprocessing.cpu_count()
    print(f"CPU Cores: {num_cpu}")
    reco_tuples = Parallel(n_jobs=num_cpu)(delayed(run_trajectory)(row, B_) for i,row in tqdm(enumerate(ic_df.itertuples()), file=sys.stdout, desc='particle #', total=len(ic_df)))
    tand0s = np.array([i[0] for i in reco_tuples])
    tands = np.array([i[1] for i in reco_tuples])
    Rmaxs = np.array([i[2] for i in reco_tuples])
    # store results in IC dataframe
    ic_df.loc[:, f"tand0_{name}"] = tand0s
    ic_df.loc[:, f"tand_{name}"] = tands
    ic_df.loc[:, f"Rmax_{name}"] = Rmaxs
    stop = time.time()
    dt = stop - start
    N = len(ic_df)
    print("Calculations Complete")
    print(f"Runtime: {dt} s, {dt/60.} min, {dt/60./60.} hr")
    print(f"Speed: {dt / (2*N)} s / trajectory\n")

    return ic_df


if __name__=='__main__':
    # Number of particles to generate
    N_gen = 10000 # 64
    # Bmaps to use and their names
    Bmaps = [ddir+'Bmaps/'+f for f in ['Mau9/Mu2e_DSMap_Mau9.p', 'Mau10/Mu2e_DSMap_Mau10.p', 'Mau12/DSMap_Mau12.p', 'Mu2e_DSMap_V13.p']]
    names = [f'Mau{i}' for i in [9, 10, 12, 13]]
    # generate initial conditions
    df = generate_IC_df(N_gen=N_gen)
    # run analysis loop for each map
    for Bmap, name in zip(Bmaps, names):
        df = map_run(df, Bmap, name)
    # save pickle
    df.to_pickle('/home/ckampa/data/pickles/emtracks/pion_degrader_tandip/degrader_tandip_df_run01.pkl')
