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
from emtracks.tools import InitConds, Bounds, ic_Mu2e, bounds_Mu2e
from emtracks.mapinterp import get_df_interp_func
from emtracks.Bdist import get_B_df_distorted

ddir = "/home/ckampa/data/"
forig = ddir+"root/Mu2e_ConversionElectron_MC.pkl"
fBnom = ddir+"Bmaps/Mu2e_DSMap_V13.p"

df_Mu2e_nom = pd.read_pickle(fBnom)
df_Mu2e_dis = get_B_df_distorted(df_Mu2e_nom, v="0")
# df_Mu2e_dis = get_B_df_distorted(df_Mu2e_nom, v="0", Bz0=1000.) # 10x gradient

# B functions
B_Mu2e_nom = get_df_interp_func(df=df_Mu2e_nom, gauss=False)#, bounds=bounds_Mu2e)
B_Mu2e_dis = get_df_interp_func(df=df_Mu2e_dis, gauss=False)#, bounds=bounds_Mu2e)

del(df_Mu2e_nom)
del(df_Mu2e_dis)

def load_origins(filename=forig, N=10):
    df = pd.read_pickle(filename)
    df = df.sample(N)
    df.reset_index(inplace=True)
    return df

def create_init_conds(origin_tuple):
    ic = InitConds(t0=ic_Mu2e.t0, tf=ic_Mu2e.tf, N_t=ic_Mu2e.N_t,
                   x0=origin_tuple.x0, y0=origin_tuple.y0, z0=origin_tuple.z0,
                   p0=origin_tuple.p_mc, theta0=origin_tuple.theta0, phi0=origin_tuple.phi0)
    return ic

def run_solver(row, fname, atol=1e-8, rtol=1e-8, verbose=False):
    ic = create_init_conds(row)
    # create solver with nominal field
    e_solver = trajectory_solver(ic, B_func=B_Mu2e_nom, bounds=bounds_Mu2e)
    # calculate true trajectory
    sol = e_solver.solve_trajectory(verbose=verbose, atol=atol, rtol=rtol)
    # reco with distorted field
    e_solver.analyze_trajectory(step=25, stride=1, query="z >= 8.41 & z <= 11.66", B=B_Mu2e_dis) # analyze in tracker region
    # save to pickle
    e_solver.to_pickle(fname+'.dis.pkl')
    df_dis = e_solver.df_reco
    # reco again with distorted field
    e_solver.analyze_trajectory(step=25, stride=1, query="z >= 8.41 & z <= 11.66", B=B_Mu2e_nom) # analyze in tracker region
    # save to pickle
    e_solver.to_pickle(fname+'.nom.pkl')
    df_nom = e_solver.df_reco
    # trajectory_solver.to_pickle(e_solver, fname)
    # return e_solver
    # run 1 and 2, entrance to tracker
    ##return df_nom.p.iloc[0], df_nom.m.iloc[0], df_dis.p.iloc[0], df_dis.m.iloc[0]
    # run 3, mean
    return df_nom.p.mean(), df_nom.m.mean(), df_dis.p.mean(), df_dis.m.mean()
    # run 4, mean, std
    return df_nom.p.mean(), df_nom.p.std(), df_dis.p.mean(), df_dis.m.std()

def run(name="test_01", outdir="/home/ckampa/data/pickles/distortions/linear_gradient/", N=50, verbose=False):
    print("Start of Distortion Simulation / Analysis")
    print("-----------------------------------------")
    print(f"Directory: {outdir},\nFilename: {name},\nNumber: {N}")
    start = time.time()
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    df_sample = load_origins(N=N)
    df_sample.to_pickle(outdir+'MC_sample.pkl')
    num_cpu = multiprocessing.cpu_count()
    print(f"CPU Cores: {num_cpu}")
    base = outdir+name
    nz = int(np.floor(np.log10(N)))
    reco_tuples = Parallel(n_jobs=num_cpu)(delayed(run_solver)(row, base+f'.{i:0{nz}d}.pkl', verbose=verbose) for i, row in tqdm(enumerate(df_sample.itertuples()), file=sys.stdout, desc='sample #'))
    p_noms = np.array([i[0] for i in reco_tuples])
    ps_noms = np.array([i[1] for i in reco_tuples])
    p_diss = np.array([i[2] for i in reco_tuples])
    ps_diss = np.array([i[3] for i in reco_tuples])
    df_sample.loc[:, "p_nom"] = p_noms
    df_sample.loc[:, "std_p_nom"] = ps_noms
    df_sample.loc[:, "p_dis"] = p_diss
    df_sample.loc[:, "std_p_dis"] = ps_diss
    df_sample.to_pickle(outdir+'MC_sample_plus_reco.pkl')
    stop = time.time()
    print("Calculations Complete")
    print(f"Runtime: {stop-start} s, {(stop-start)/60.} min, {(stop-start)/60./60.} hr")
    print(f"Speed: {(stop-start) / (2*N)} s / trajectory")
    return
    # return noms, diss

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
    if args.number is None:
        args.number = 10
    else:
        args.number = int(args.number)
    run(name=args.filename, outdir=args.directory, N=args.number)
