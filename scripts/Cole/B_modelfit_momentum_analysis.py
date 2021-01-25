# now using LHelix reconstruction
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

# from emtracks.impact_param import *
from emtracks.particle import trajectory_solver
from emtracks.mapinterp import get_df_interp_func
from emtracks.Bdist import get_B_df_distorted

# directories
# datadir = '/home/ckampa/data/pickles/distortions/linear_gradient/'
datadir = "/home/ckampa/data/pickles/emtracks/mid_hp_bias_up/"
# set up B interpolators
ddir = '/home/ckampa/data/'
# ddir = '/home/shared_data/'
# Test of Brian Fit
# fBnom = ddir+"Bmaps/Mu2e_DSMap_V13.p"
# fBdis = ddir+"Bmaps/Mau13_standard_tracker_fit_df.p"
# Test of Hall Probe biased up
fBnom = ddir+"Bmaps/Mau13_standard_tracker_fit_df.p"
# fBdis = ddir+"Bmaps/Mau13_middle_hp_bias_up_fit_df.p"
nums = [0, 1, 2, 3, 4]
fBdis_list = [ddir+f"Bmaps/hp_bias/Mau13_hp_{num}_bias_up_fit_df.p" for num in nums]

# get interp funcs
B_Mu2e_nom = get_df_interp_func(fBnom, gauss=False, Blabels=['Bx_fit', 'By_fit', 'Bz_fit'])
# B_Mu2e_dis = get_df_interp_func(fBdis, gauss=False, Blabels=['Bx_fit', 'By_fit', 'Bz_fit'])
B_Mu2e_dis_list = [get_df_interp_func(fBdis, gauss=False, Blabels=['Bx_fit', 'By_fit', 'Bz_fit']) for fBdis in fBdis_list]

step=100 # 100 default

# analyze a given track w nominal and distorted field
def analyze_particle_momentum(particle_num, name, outdir):
    # load track (pickle)
    fname = outdir+name+f'.{particle_num:03d}.pkl.nom.pkl'
    e_Mu2e = trajectory_solver.from_pickle(fname)
    # analyze
    # nominal
    e_Mu2e.B_func = B_Mu2e_nom
    e_Mu2e.analyze_trajectory_LHelix(step=step, stride=1)
    mom_nom = e_Mu2e.mom_LHelix
    # distorted
    mom_dis_list = []
    for B_Mu2e_dis in B_Mu2e_dis_list:
        e_Mu2e.B_func = B_Mu2e_dis
        e_Mu2e.analyze_trajectory_LHelix(step=step, stride=1)
        mom_dis_list.append(e_Mu2e.mom_LHelix)
    return mom_nom, mom_dis_list

# driving function
def run_analysis(name="run_04", outdir="/home/ckampa/data/pickles/distortions/linear_gradient/run_04/", N_lim=None):
    print("Start of Distortion LHelix Momentum Analysis")
    print("--------------------------------------------")
    print(f"Directory: {outdir},\nFilename: {name}")
    start = time.time()
    df_run = pd.read_pickle(outdir+'MC_sample_plus_reco.pkl')
    num_cpu = multiprocessing.cpu_count()
    print(f"CPU Cores: {num_cpu}")
    base = outdir+name
    # particle_nums = [int(f[7:10]) for f in sorted(os.listdir(outdir)) if "nom" in f]
    particle_nums = [int(f[f.index('.')+1:f.index('.')+1+3]) for f in sorted(os.listdir(outdir)) if "nom" in f]
    if N_lim is not None:
        particle_nums = particle_nums[:int(N_lim)]
        df_run = df_run.iloc[:int(N_lim)].copy()
        df_run.reset_index(drop=True, inplace=True)
    N = len(df_run)
    reco_tuples = Parallel(n_jobs=num_cpu)(delayed(analyze_particle_momentum)(num, name=name, outdir=outdir) for num in tqdm(particle_nums, file=sys.stdout, desc='particle #'))
    mom_noms = np.array([i[0] for i in reco_tuples])
    mom_diss = np.array([i[1] for i in reco_tuples])
    results_dict = {f'mom_dis_hp_{num}': mom_diss[:,i] for i,num in enumerate(nums)}
    results_dict['mom_nom'] = mom_noms
    # save to dataframe
    for key, val in results_dict.items():
        df_run.loc[:, key] = val
    ######
    # results_dict = {'mom_nom':mom_noms, 'mom_dis_fit':mom_diss}
    # save to dataframe
    # for key, val in results_dict.items():
    #     df_run.loc[:, key] = val
    # df_run.to_pickle(outdir+'MC_sample_plus_reco_Mau13_ModelFit.pkl') # DS Solenoid Off (Mau13 subtraction)
    # df_run.to_pickle(outdir+'MC_sample_plus_reco_mid_hp_bias_up.pkl') # Biased Hall probe
    df_run.to_pickle(outdir+'MC_sample_plus_reco_all_hp_bias_up.pkl') # Biased Hall probe
    stop = time.time()
    print("Calculations Complete")
    print(f"Runtime: {stop-start} s, {(stop-start)/60.} min, {(stop-start)/60./60.} hr")
    print(f"Speed: {(stop-start) / (2*N)} s / trajectory")
    return

if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', help='Output directory')
    parser.add_argument('-f', '--filename', help='Output base filename')
    parser.add_argument('-N', '--number', help='Number of events')
    args = parser.parse_args()
    # fill defaults where needed
    if args.directory is None:
        args.directory = "/home/ckampa/data/pickles/distortions/linear_gradient/testing/"
    if args.filename is None:
        args.filename = "test_01"
    # run
    run_analysis(name=args.filename, outdir=args.directory, N_lim=args.number)
