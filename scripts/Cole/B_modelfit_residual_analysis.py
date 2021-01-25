import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt

from emtracks.particle import trajectory_solver
from emtracks.mapinterp import get_df_interp_func
from emtracks.plotting import config_plots

config_plots()
plt.rcParams['figure.figsize'] = [13, 11] # larger figures

# load and prep tracker fit dataframe
# df_fit = pd.read_pickle('/home/ckampa/data/Bmaps/Mau13_standard_tracker_fit_df.p')
# df_fit = pd.read_pickle('/home/ckampa/data/Bmaps/Mau13_middle_hp_bias_up_fit_df.p')
# nums = [0, 1, 2, 3, 4]
num = 4
df_fit = pd.read_pickle(f'/home/ckampa/data/Bmaps/hp_bias/Mau13_hp_{num}_bias_up_fit_df.p')
df_fit.eval('B = (Bx**2+By**2+Bz**2)**(1/2)', inplace=True)
df_fit.eval('B_fit = (Bx_fit**2+By_fit**2+Bz_fit**2)**(1/2)', inplace=True)
df_fit.eval('B_res = B - B_fit', inplace=True)
df_fit.eval('Bx_res = Bx - Bx_fit', inplace=True)
df_fit.eval('By_res = By - By_fit', inplace=True)
df_fit.eval('Bz_res = Bz - Bz_fit', inplace=True)
# setup interp functions
xyz_res_func = get_df_interp_func(df=df_fit, Blabels=['Bx_res','By_res','Bz_res'])
mag_res_func = get_df_interp_func(df=df_fit, Blabels=['B_res','By_res','Bz_res'])


# RUN = "run_04"
# E_ANALYSIS = False
# EMTRACK_RUN_DIR = f"/home/ckampa/data/pickles/distortions/linear_gradient/{RUN}/"
# datadir = f"/home/ckampa/data/pickles/distortions/linear_gradient/"
# ESTRIDE = 10 # testing (3 cm)
ESTRIDE = 1 # real (3 mm)

def track_residual(particle_num, Rrange=None, name='run_04', outdir='/home/ckampa/data/pickles/distortions/linear_gradient/run_04/'):
    # tracker radius range is about 40cm < r < 70cm
    query = 'z >= 8.41 & z <= 11.66' # tracker query
    fname = outdir+name+f'.{particle_num:03d}.pkl.nom.pkl'
    e = trajectory_solver.from_pickle(fname)
    if not Rrange is None:
        query += f'& r >= {Rrange[0]} & r <= {Rrange[1]}'
        e.dataframe.eval('r = (x**2+y**2)**(1/2)', inplace=True)
    xyz_track = e.dataframe.query(query)[['x','y','z']].values[::ESTRIDE]
    xyz_res_track = np.array([xyz_res_func(xyz) for xyz in xyz_track])
    mag_res_track = np.array([mag_res_func(xyz) for xyz in xyz_track])[:,0].reshape(-1, 1)
    # return np.concatenate([xyz_res_track, mag_res_track], axis=1)
    return np.concatenate([xyz_res_track, mag_res_track], axis=1)

# driving function
def run_analysis(name="run_04", outdir="/home/ckampa/data/pickles/distortions/linear_gradient/run_04/", N_lim=None):
    print("Start of Fit Residual Track Analysis")
    print("--------------------------------------------")
    print(f"Directory: {outdir},\nFilename: {name}")
    start = time.time()
    # df_run = pd.read_pickle(outdir+'MC_sample_plus_reco.pkl')
    num_cpu = multiprocessing.cpu_count()
    print(f"CPU Cores: {num_cpu}")
    base = outdir+name
    # particle_nums = [int(f[7:10]) for f in sorted(os.listdir(datadir+'run_04/')) if "nom" in f]
    particle_nums = [int(f[f.index('.')+1:f.index('.')+1+3]) for f in sorted(os.listdir(outdir)) if "nom" in f]
    if N_lim is not None:
        particle_nums = particle_nums[:int(N_lim)]
        # df_run = df_run.iloc[:int(N_lim)].copy()
        # df_run.reset_index(drop=True, inplace=True)
    # N = len(df_run)
    N = len(particle_nums)
    reco_tuples = Parallel(n_jobs=num_cpu)(delayed(track_residual)(num, Rrange=None, name=name, outdir=outdir) for num in tqdm(particle_nums, file=sys.stdout, desc='particle #'))
    # reco_tuples = Parallel(n_jobs=num_cpu)(delayed(track_residual)(num, Rrange=[.4,.7], name=name, outdir=outdir) for num in tqdm(particle_nums, file=sys.stdout, desc='particle #'))
    Bx_ress = np.concatenate([i[:,0] for i in reco_tuples])#.flatten()
    By_ress = np.concatenate([i[:,1] for i in reco_tuples])#.flatten()
    Bz_ress = np.concatenate([i[:,2] for i in reco_tuples])#.flatten()
    B_ress = np.concatenate([i[:,3] for i in reco_tuples])#.flatten()
    print(Bx_ress.shape, By_ress.shape, Bz_ress.shape, B_ress.shape)
    results_dict = {'Bx_res':Bx_ress, 'By_res':By_ress, 'Bz_res':Bz_ress, 'B_res':B_ress}
    # mom_noms = np.array([i[0] for i in reco_tuples])
    # mom_diss = np.array([i[1] for i in reco_tuples])
    # results_dict = {'mom_nom':mom_noms, 'mom_dis_fit':mom_diss}
    # save to dataframe
    # for key, val in results_dict.items():
    #     df_run.loc[:, key] = val
    df_run = pd.DataFrame(results_dict)
    # df_run.to_pickle(outdir+'B_Residuals_Mau13_Fit.pkl')
    # df_run.to_pickle(outdir+'B_Residuals_Mau13_Fit_Straws.pkl')
    df_run.to_pickle(outdir+f'B_Residuals_Mau13_Fit_hp_{num}.pkl')
    # df_run.to_pickle(outdir+f'B_Residuals_Mau13_Fit_hp_{num}_Straws.pkl')
    stop = time.time()
    print("Calculations Complete")
    print(f"Runtime: {stop-start} s, {(stop-start)/60.} min, {(stop-start)/60./60.} hr")
    print(f"Speed: {(stop-start) / (2*N)} s / trajectory")
    return

if __name__=='__main__':
    # run_analysis() # nominal run checking Brian fit
    # run_analysis(name="mid_hp_bias_up", outdir='/home/ckampa/data/pickles/emtracks/mid_hp_bias_up/') # check biased Hall probe
    run_analysis(name="mid_hp_bias_up", outdir='/home/ckampa/data/pickles/emtracks/mid_hp_bias_up/') # check biased Hall probe
    # run_analysis(N_lim=100)
    # parse command line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--directory', help='Output directory')
    # parser.add_argument('-f', '--filename', help='Output base filename')
    # parser.add_argument('-N', '--number', help='Number of events')
    # args = parser.parse_args()
    # # fill defaults where needed
    # if args.directory is None:
    #     args.directory = "/home/ckampa/data/pickles/distortions/linear_gradient/testing/"
    # if args.filename is None:
    #     args.filename = "test_01"
    # run
    # run_analysis(name=args.filename, outdir=args.directory, N_lim=args.number)
