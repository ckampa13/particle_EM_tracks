# for analyzing the 50 x 50 x 11 (theta x phi x TS scale) (all same x0, y0, z0) that Darren generated
# using LHelix momentum reco
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

from emtracks.particle import trajectory_solver
from emtracks.mapinterp import get_df_interp_func

# generated momentum
p0 = 104.96 # MeV/c

# directories
datadir = '/home/shared_data/mao10,mao13_analysis/data/mao13no_nom/'

# set up B interpolator
ddir = '/home/shared_data/'
fBnom = ddir+"Bmaps/Mu2e_DSMap_V13.p"
B_Mu2e_nom = get_df_interp_func(fBnom, gauss=False)

# analyze a given track w nominal and distorted field
def analyze_particle_momentum(filename, datadir=datadir):
    # load track (pickle)
    # fname = datadir+f'{scale}_{theta}_{phi}_0.054_.pkl'
    e = trajectory_solver.from_pickle(datadir+filename)
    # scale from filename
    scale = float(filename[:4])
    # get full theta, phi from B
    theta0 = e.init_conds.theta0
    phi0 = e.init_conds.phi0
    # analyze in nominal only
    # nominal
    e.B_func = B_Mu2e_nom
    e.analyze_trajectory_LHelix(step=50, stride=1)
    mom_nom = e.mom_LHelix
    status = e.LHelix_success
    return scale, theta0, phi0, mom_nom, status

# driving function
def run_analysis(N_lim=None):
    print("Start of Distortion LHelix Momentum Analysis")
    print("---------------------------___--------------")
    print(f"Directory: {datadir},\nFilename: TSSCALE_THETA0_PHI0_X0_NAME.pkl")
    start = time.time()
    num_cpu = multiprocessing.cpu_count()
    print(f"CPU Cores: {num_cpu}")
    files = os.listdir(datadir)
    if N_lim is not None:
        files = files[:int(N_lim)]
    N = len(files)
    reco_tuples = Parallel(n_jobs=num_cpu)(delayed(analyze_particle_momentum)(filename) for filename in tqdm(files, file=sys.stdout, desc='track #'))
    scales = np.array([i[0] for i in reco_tuples])
    thetas = np.array([i[1] for i in reco_tuples])
    phis = np.array([i[2] for i in reco_tuples])
    moms = np.array([i[3] for i in reco_tuples])
    statuss = np.array([i[4] for i in reco_tuples])
    p0s = p0 * np.ones_like(scales)
    results_dict = {'mom_nom': moms, 'TS_scale':scales, 'theta0':thetas, 'phi0':phis, 'status':statuss, 'mom_MC': p0s}
    df_run = pd.DataFrame(results_dict)
    df_run.to_pickle('/home/ckampa/data/pickles/distortions/darren/MC_mom_reco_TS_Scaled_Mau13.pkl')
    stop = time.time()
    print("Calculations Complete")
    print(f"Runtime: {stop-start} s, {(stop-start)/60.} min, {(stop-start)/60./60.} hr")
    print(f"Speed: {(stop-start) / (2*N)} s / trajectory")
    return df_run

if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--number', help='Number of events')
    args = parser.parse_args()
    # run
    run_analysis(N_lim=args.number)
