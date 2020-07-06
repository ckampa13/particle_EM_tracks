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
import os

testdir = "/home/darren/Desktop/plots/"
datadir = "/home/shared_data/"
plotdir = datadir+"plots/randomphi/"
mapdir = datadir+"Bmaps/"
date = "/6-20/"
newdir = datadir+'50_Gauss_z_3-14'

#0.0 vs 0.0605 etc, bins in histogram
def readpickle():
    files = sorted(os.listdir(newdir))
    files[0], files[1], files[2], files[3] = files[2], files[3], files[0], files[1]

    phinom = []
    phidis = []
    nomdata = []
    disdata = []

    nom = {}
    dis = {}

    for file in files:
        if file.endswith('nom.pkl'):
            e_solvernom = trajectory_solver.from_pickle(newdir+'/'+file)
            phinom.append(e_solvernom.init_conds.phi0)
            nomdata.append(e_solvernom.dataframe)
            #nom.update({str(e_solvernom.init_conds.phi0), e_solvernom.dataframe})
            
        if file.endswith('dis.pkl'):
            e_solverdis = trajectory_solver.from_pickle(newdir+'/'+file)
            phidis.append(e_solverdis.init_conds.phi0)
            disdata.append(e_solverdis.dataframe)
            #dis.update({str(e_solverdis.init_conds.phi0), e_solverdis.dataframe})
    if phinom == phidis:
        for i in range(0, len(phinom), 1):
            nom.update({phinom[i]:nomdata[i]})
            dis.update({phidis[i]:disdata[i]})
            
    if phinom == phidis:
        return nom, dis, phinom, len(phinom)

def find_track_at_z(df, z): 
	delta = (df.z.max() - df.z.min()) / len(df.z)
	#delta = 10/4001   #approximate z range divided by number of points
	mask = (df['z'] < z + delta) & (df['z'] > z - delta)
	
	while (len(df.z[mask]) > 2):
		delta = delta / 2
		mask = (df.z < z + delta) & (df.z > z - delta)
	while (len(df.z[mask]) == 0):
		delta = delta*2
		mask = (df.z < z + delta) & (df.z > z - delta)
	
	if len(df.z[mask]) == 1:
		df2 = df.loc[mask]
		df2 = df2.apply(pd.to_numeric)
		return ([df2.iloc[0]['x'], df2.iloc[0]['y'], df2.iloc[0]['z'], df2.iloc[0]['t'], df2.iloc[0]['r']])

	if len(df.z[mask]) == 2:
		df2 = df.loc[mask]
		df2 = df2.apply(pd.to_numeric)
	
		x1 = df2.iloc[0]['x']
		x2 = df2.iloc[1]['x']
		y1 = df2.iloc[0]['y']
		y2 = df2.iloc[1]['y']
		t1 = df2.iloc[0]['t']
		t2 = df2.iloc[1]['t']
		z1 = df2.iloc[0]['z']
		z2 = df2.iloc[1]['z']

		xslope = (x2-x1)/(z2-z1)
		yslope = (y2-y1)/(z2-z1)
		tslope = (t2-t1)/(z2-z1)

		xinterp = x2 - ((z2-z)*(xslope))
		yinterp = y2 - ((z2-z)*(yslope))
		tinterp = t2 - ((z2-z)*(tslope))
		rinterp = (xinterp**2 + yinterp**2)**(1/2)

		return (xinterp, yinterp, z, tinterp, rinterp)

def restructure1(phis, nomdata, disdata, z):
	nts = []
	nxs = []
	nys = []
	nzs = []
	nrs = []
	
	dts = []
	dxs = []
	dys = []
	dzs = []
	drs = []

	for i in range (0, len(nomdata), 1): #len(nomdata) = #of phis
		nx, ny, nz, nt, nr = find_track_at_z(nomdata[(phis[i])], z)
		dx, dy, dz, dt, dr = find_track_at_z(disdata[(phis[i])], z)
		
		nts.append(nt)
		nxs.append(nx)
		nys.append(ny)
		nzs.append(nz)
		nrs.append(nr)
		
		dts.append(dt)
		dxs.append(dx)
		dys.append(dy)
		dzs.append(dz)
		drs.append(dr)
		   
	nts = np.array(nts)
	nxs = np.array(nxs)
	nys = np.array(nys)
	nzs = np.array(nzs)
	nrs = np.array(nrs)
	
	dts = np.array(dts)
	dxs = np.array(dxs)
	dys = np.array(dys)
	dzs = np.array(dzs)
	drs = np.array(drs)
	
	x = (nts, phis, nrs, nxs, nys, nzs)
	y = (dts, phis, drs, dxs, dys, dzs)
	
	return x, y

def restructure2(zstart, zend, numpoints, nomdata, disdata, phis, N):   #rmax, rmin, xmax, xmin, ymax, ymin
	q = np.linspace(zstart, zend, numpoints) #all the z values u need in a list
	e = np.tile(q, len(phis)) #repeat q len(phis) times in the same list
	f = np.repeat(phis, numpoints) #phis list repeated numpoints times
	
	nomfinaldata = []
	disfinaldata = []
	
	#construct the data in two different ways (seems inefficient by looping this many times)
	for phi in phis:
		for i in q:
			nomfinaldata.append(find_track_at_z(nomdata[(phi)], i))
			disfinaldata.append(find_track_at_z(disdata[(phi)], i))
	
	nomfinaldatares = []
	disfinaldatares = []
	 
	for i in q:
		transitionlistnom = []
		transitionlistdis = []
		
		for phi in phis:
			transitionlistnom.append(find_track_at_z(nomdata[(phi)], i))
			transitionlistdis.append(find_track_at_z(disdata[(phi)], i))
		
		nomfinaldatares.append(transitionlistnom)
		disfinaldatares.append(transitionlistdis)
	
	#construct dataframes
	arrays = (f, e)
	tuples = list(zip(*arrays))
	index = pd.MultiIndex.from_tuples(tuples, names = ['phi', 'z_val'])
	
	df1 = pd.DataFrame(nomfinaldata, columns = ["X", "Y", "Z", "T", "R"], index = index)
	df2 = df1.stack().unstack(0)
	
	return nomfinaldatares, disfinaldatares, e, f, q, numpoints, df1, df2

def plothist(x, y):
	fig1 = plt.figure()
	x1, x2 = x[2], y[2]
	num_bins = 20
	x1num_bins = int((max(x1) - min(x1)) / 0.005)
	x2num_bins = int((max(x2) - min(x2)) / 0.005)

	plt.hist(x[2], alpha = 0.3, bins = x1num_bins, color='blue', label = 'Mu2e Field')
	plt.hist(y[2], alpha = 0.8, bins = x2num_bins, color='orange', label = 'Graded Field')
	plt.xlabel('R (meters)')
	plt.ylabel('Occurences')
	plt.title('Histogram of X At Z=13 Meters with Varying Initial Phi')  
	#plt.show()   

	fig2 = plt.figure()
	x1, x2 = x[3], y[3]
	num_bins = 20
	x1num_bins = int((max(x1) - min(x1)) / 0.005)
	x2num_bins = int((max(x2) - min(x2)) / 0.005)

	plt.hist(x[2], alpha = 0.3, bins = x1num_bins, facecolor='blue', label = 'Mu2e Field')
	plt.hist(y[2], alpha = 0.8, bins = x2num_bins, facecolor='orange', label = 'Graded Field')
	plt.xlabel('X (meters)')
	plt.ylabel('Occurences')
	plt.title('Histogram of Radius At Z=13 Meters, varying initial phi')  
	#plt.show() 
	
	fig3 = plt.figure()
	x1, x2 = x[4], y[4]
	num_bins = 20
	x1num_bins = int((max(x1) - min(x1)) / 0.005)
	x2num_bins = int((max(x2) - min(x2)) / 0.005)

	plt.hist(x[2], alpha = 0.3, bins = x1num_bins, facecolor='blue', label = 'Mu2e Field')
	plt.hist(y[2], alpha = 0.8, bins = x2num_bins, facecolor='orange', label = 'Graded Field')
	plt.xlabel('Y (meters)')
	plt.ylabel('Occurences')
	plt.title('Histogram of Y At Z=13 Meters, varying initial phi')  
	#plt.show()   
	return fig1, fig2, fig3

def plotscatter(x, y): #(ts, phis, rs, xs, ys, zs)
	phis = x[1]
	xs = x[2]
	ys = y[2]

	fig1 = plt.figure()
	plt.scatter(phis, xs, c=x[1], marker='s', label='nom')
	plt.scatter(phis, ys, c=x[1], marker='o', label ='dis')
	plt.legend(loc='upper right')
	plt.xlabel("phis")
	plt.ylabel("radius (meters)")
	plt.title("Radius at z=13 for Nominal and Graded Tracks with Varying Initial Phi")
	cbar = plt.colorbar()
	cbar.set_label('phis')
	#plt.show()

	dev = y[2]-x[2]
	fig2 = plt.figure()
	plt.scatter(phis, dev, c=x[1])
	plt.xlabel("phis")
	plt.ylabel("radius (meters)")
	plt.title("Radius Displacement at z=13 Between Nom and Dis Tracks with Varying Initial Phi")
	#plt.show()

	return fig1, fig2


def plotdifferences(phis, q, nomfinaldatares, disfinaldatares):
	title = phis
	title.insert(0, 'z')
	xdif = [title]
	ydif = [title]
	rdif = [title]

	for i in range(0, len(nomfinaldatares), 1): #specific z value
		xtransitionlist = [q[i]] #all the z points u want
		ytransitionlist = [q[i]]
		rtransitionlist = [q[i]]
	
		for j in range(0, len(nomfinaldatares[0])): #specific phi value
			xtransitionlist.append(nomfinaldatares[i][j][0] - disfinaldatares[i][j][0])
			ytransitionlist.append(nomfinaldatares[i][j][1] - disfinaldatares[i][j][1])
			rtransitionlist.append(nomfinaldatares[i][j][4] - disfinaldatares[i][j][4])
	
		xdif.append(xtransitionlist)
		ydif.append(ytransitionlist)
		rdif.append(rtransitionlist)  #rdif[0] - has same z values, rdif[0][1] has same phi value, 
									  #rdif[i][0] when i > 0 is the z value for that block
									  #phi values top row
	fig = plt.figure()
	for i in range(1, len(rdif), 1):
		a = np.full((len(rdif[1])-1), rdif[i][0])
		plt.scatter(a, rdif[i][1:], c = rdif[0][1:])

	 
	plt.xlabel("z (meters)")
	plt.ylabel("r_deviations (meters)")
	plt.title("Radial Deviations for 15 Particle Tracks at Z due to 50 Gauss Linear Graded Bdist (blue-0, red-pi)")
	cbar = plt.colorbar()
	cbar.set_label('phis')
	#plt.show()
	#fig.savefig(testdir+'6-22-7')

	fig1 = plt.figure()
	for i in range(1, len(xdif), 1):
		a = np.full((len(xdif[1])-1), xdif[i][0])
		plt.scatter(a, xdif[i][1:], c = xdif[0][1:])
	
	plt.xlabel("z (meters)")
	plt.ylabel("x_deviations (meters)")
	plt.title("X Deviations for 15 Particle Tracks at Z in MU2E Detector due to 50 Gauss Linear Graded Bdist")
	cbar = plt.colorbar()
	cbar.set_label('phis')
	#plt.show()
	#fig1.savefig(testdir+'6-22-8')

	fig2 = plt.figure()
	for i in range(1, len(ydif), 1):
		a = np.full((len(ydif[1])-1), ydif[i][0])
		plt.scatter(a, ydif[i][1:], c = ydif[0][1:])
	
	plt.xlabel("z (meters)")
	plt.ylabel("y_deviations (meters)")
	plt.title("Y Deviations for 15 Particle Tracks at Z in MU2E Detector due to 50 Gauss Linear Graded Bdist")
	cbar = plt.colorbar()
	cbar.set_label('phis')
	#plt.show()
	#fig2.savefig(testdir+'6-22-9')

	return fig, fig1, fig2


if __name__ == '__main__':
	nomdata, disdata, phis, N = readpickle()
	x, y = restructure1(phis, nomdata, disdata, 13)
	
	plothist(x, y)
	plotscatter(x, y)

	z = restructure2(6, 13, 20, nomdata, disdata, phis, N)
	df1 = z[6]
	df2 = z[7]
	nomfinaldatares = np.array(z[0])
	disfinaldatares = np.array(z[1])
	q = (z[4])

	plotdifferences(phis, q, nomfinaldatares, disfinaldatares)
	
	plt.show()



def yvsx(phis, nomdata, disdata, z):
	nom, dis = restructure1(phis, nomdata, disdata, z)
	fig1 = plt.figure()
	plt.scatter(nom[3], nom[4], c='b', marker='s', label='nom')
	plt.scatter(dis[3], dis[4], c='r', marker='s', label='dis')
	plt.title(f'Y vs X for Nominal and Distorted Fields at Z = {z}')
	plt.legend(loc='upper right')
	plt.xlabel("X (meters)")
	plt.ylabel("Y (meters)")

def res(z, phis, nomdata, disdata):
	nom, dis = restructure1(phis, nomdata, disdata, z)
	return nom[3], nom[4], dis[3], dis[4]

def plot3d(phis, nomdata, disdata, zstart, zend, numsteps):
	q = np.linspace(zstart, zend, numsteps) #all the z values u need in a list
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for i in q:
		ax.scatter(res(i,phis,nomdata,disdata)[0], res(i,phis, nomdata, disdata)[1], i, c='b')
		ax.scatter(res(i,phis,nomdata,disdata)[2], res(i,phis, nomdata, disdata)[3], i, c='r')

if __name__ == '_3_main__':
	nomdata, disdata, phis, N = readpickle()
	yvsx(phis, nomdata, disdata, 11)
	plot3d(phis, nomdata, disdata, 6, 13, 100)
	plt.show()
	










	





