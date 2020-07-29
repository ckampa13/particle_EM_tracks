# import package
# installed via pip
from emtracks.particle import * # main solver object
from emtracks.conversions import one_gev_c2_to_kg # conversion for q factor (transverse momentum estimate)
from emtracks.tools import *#InitConds # initial conditions namedtuple
from emtracks.mapinterp import get_df_interp_func  # factory function for creating Mu2e DS interpolation function
from emtracks.Bdist import get_B_df_distorted
from emtracks.interpolations import *
import matplotlib.animation as animation
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
import os

from joblib import Parallel, delayed
import multiprocessing
from tqdm.notebook import tqdm


rad13plotdir = '/home/shared_data/mao10,mao13_analysis/plots/mao13(0.90,1.10TS)rad/'
reg13plotdir = '/home/shared_data/mao10,mao13_analysis/plots/mao13(0.90,1.10TS)/'
mao13datadir = '/home/shared_data/mao10,mao13_analysis/data/mao13(0.90,1.10TS)/'

def readpklold(zstart, zend, numpoints):
    files = sorted(os.listdir(mao13datadir))
    zsteps = np.linspace(zstart, zend, numpoints)
    data = []
    deleted = []

    for file in files:
        x = file.split('_')
        field = x[0]
        e_solvernom = trajectory_solver.from_pickle(mao13datadir+file)
        phi = e_solvernom.init_conds.phi0
        theta = e_solvernom.init_conds.theta0
        for z in zsteps:
            if z > e_solvernom.dataframe.z.max() or z < e_solvernom.dataframe.z.min():
                data.append(8*[np.nan])
                deleted.append([e_solvernom.init_conds.theta0, e_solvernom.init_conds.phi0])
            else:
                info = interp_cole(e_solvernom.dataframe, z)
                x = info[1]
                y = info[2]
                r = tuple([(x**2 + y**2)**(1/2)])
                r2 = tuple([((x-0.054094482)**(2) + (y-0.03873037)**(2))**(1/2)])
                tuple1 = (z, field, theta, phi)
                
                nan = tuple([np.nan])

                data.append(tuple1 + info + r + r2)
                
    return data, deleted

def difarrays(df):
    ms = []
    data = []
    missing = []
    for z in df['z'].unique():
        for field in df['field'].unique():
            xdifs = []
            ydifs = []
            rdifs = []
            for theta in df['theta'].unique():
                dfnew1 = df[(df['z']==z) & np.isclose(df['theta'], theta, 1e-1) & ((df['field']=='1.00'))]
                dfnew2 = df[(df['z']==z) & np.isclose(df['theta'], theta, 1e-1) & ((df['field']==field))]
                if len(dfnew1['theta'].unique()) > 1:
                    return ('problem with tolerance')
                if len(np.array(dfnew1['phi'])) != len(np.array(dfnew2['phi'])):
                    missing.append([z, field, theta])
                elif(np.isclose(v, 0) for v in (np.array(dfnew1['phi']) - np.array(dfnew2['phi']))):
                    xdif = np.array(dfnew2['x']) - np.array(dfnew1['x'])
                    ydif = np.array(dfnew2['y']) - np.array(dfnew1['y'])
                    
                
                    for a in range(0, len(xdif), 1):
                        data.append([xdif[a], ydif[a], z, field, theta])
                    for a in xdif:
                        xdifs.append(a)
                    for b in ydif:
                        ydifs.append(b)
                        
                else:
                    print('mistake')
            ms.append([field, z, np.mean(xdifs), np.mean(ydifs), np.std(xdifs), np.std(ydifs)])
                
    datadf = pd.DataFrame(data, columns = ['xdif', 'ydif', 'z', 'field', 'theta'])
    msdf = pd.DataFrame(ms, columns = ['field', 'z', 'xmean', 'ymean', 'xstd', 'ystd'])
                
    return msdf, datadf, missing 
if __name__ == '__main__':
	x, deleted = readpklold(6, 13, 8)
	df = pd.DataFrame(x, columns = ['z', 'field', 'theta', 'phi', 't', 'x', 'y', 'r_(0,0)', 'r_(x0,y0)'])

	df1 = df[df['field']!='nom']
	dfnew = df1.dropna()

	msdf, datadf, missing = difarrays(dfnew)