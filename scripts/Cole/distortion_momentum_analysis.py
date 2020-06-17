import os
import sys
import argparse
import time
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

from emtracks.impact_param import *
from emtracks.mapinterp import get_df_interp_func
from emtracks.Bdist import get_B_df_distorted

# directories
datadir = '/home/ckampa/data/pickles/distortions/linear_gradient/'
plotdir = '/home/ckampa/data/plots/distortions/linear_gradient/'

# set up B interpolators
ddir = '/home/ckampa/data/'
fBnom = ddir+"Bmaps/Mu2e_DSMap_V13.p"

df_Mu2e_nom = pd.read_pickle(fBnom)
df_Mu2e_dis = get_B_df_distorted(df_Mu2e_nom, v="0")
# B functions
B_Mu2e_nom = get_df_interp_func(df=df_Mu2e_nom, gauss=False)#, bounds=bounds_Mu2e)
B_Mu2e_dis = get_df_interp_func(df=df_Mu2e_dis, gauss=False)#, bounds=bounds_Mu2e)

del(df_Mu2e_nom)
del(df_Mu2e_dis)

# for d in [plotdir, plotdir+'run_01/', plotdir+'run_02-10x_grad/']:
for d in [plotdir, plotdir+'run_04/',]:# plotdir+'run_02-10x_grad/']:
    if not os.path.exists(d):
        os.makedirs(d)

def analyze_particle_momentum(particle_num, name, outdir):
    # load track (pickle)
    fname = outdir+name+f'.{particle_num:03d}.pkl.nom.pkl'
    e_Mu2e = trajectory_solver.from_pickle(fname)
    # analyze
    # nominal
    chi2_fit_mom, circle_fit_mom = momentum_analysis_global_chi2(e_Mu2e, B_Mu2e_nom)
    # distorted
    chi2_fit_mom_dis, circle_fit_mom_dis = momentum_analysis_global_chi2(e_Mu2e, B_Mu2e_dis)
    return chi2_fit_mom, circle_fit_mom, chi2_fit_mom_dis, circle_fit_mom_dis

def run_analysis(name="run_04", outdir="/home/ckampa/data/pickles/distortions/linear_gradient/run_04/", N_lim=None):
    print("Start of Distortion Momentum Analysis")
    print("-----------------------------------------")
    print(f"Directory: {outdir},\nFilename: {name}")
    start = time.time()
    df_run = pd.read_pickle(outdir+'MC_sample_plus_reco.pkl')
    # if not os.path.exists(outdir):
    #     os.makedirs(outdir)
    # df_sample = load_origins(N=N)
    # df_sample.to_pickle(outdir+'MC_sample.pkl')
    num_cpu = multiprocessing.cpu_count()
    print(f"CPU Cores: {num_cpu}")
    base = outdir+name
    particle_nums = [int(f[7:10]) for f in sorted(os.listdir(datadir+'run_04/')) if "nom" in f]
    if N_lim is not None:
        particle_nums = particle_nums[:int(N_lim)]
        df_run = df_run.iloc[:int(N_lim)].copy()
        df_run.reset_index(drop=True, inplace=True)
    N = len(df_run)
    reco_tuples = Parallel(n_jobs=num_cpu)(delayed(analyze_particle_momentum)(num, name=name, outdir=outdir) for num in tqdm(particle_nums, file=sys.stdout, desc='particle #'))
    p_chi2_noms = np.array([i[0] for i in reco_tuples])
    p_circ_noms = np.array([i[1] for i in reco_tuples])
    p_chi2_diss = np.array([i[2] for i in reco_tuples])
    p_circ_diss = np.array([i[3] for i in reco_tuples])
    # save to dataframe
    df_run.loc[:, 'p_chi2_nom'] = p_chi2_noms
    df_run.loc[:, 'p_circ_nom'] = p_circ_noms
    df_run.loc[:, 'p_chi2_dis'] = p_chi2_diss
    df_run.loc[:, 'p_circ_dis'] = p_circ_diss
    df_run.to_pickle(outdir+'MC_sample_plus_reco_chi2.pkl')
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
    # if args.number is None:
    #     args.number = 10
    # else:
    #     args.number = int(args.number)
    run_analysis(name=args.filename, outdir=args.directory, N_lim=args.number)
