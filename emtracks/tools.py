import numpy as np
from collections import namedtuple

# InitConds = namedtuple('InitConds',['t0','tf','N_t','x0','y0','z0','p0','theta0','phi0'])
InitConds = namedtuple('InitConds',['t0','tf','N_t','x0','y0','z0','p0','theta0','phi0'])
Bounds = namedtuple('Bounds',['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'])

# a couple of canned examples
# default example
ic = InitConds(t0=0., tf=1e-8, N_t=1001, x0=0., y0=0., z0=0., p0=105., theta0=np.pi/6, phi0=0.)
bounds = Bounds(xmin=-0.95, xmax=0.95, ymin=-0.95, ymax=0.95, zmin=-1., zmax=1.)

# Mu2e examples
ic_Mu2e = InitConds(t0=0., tf=2e-7, N_t=20001,
                    x0=0.054094482, y0=0.03873037, z0=5.988900879,
                    p0=104.96, theta0=np.pi/3, phi0=0.)
ic_Mu2e_bounce = InitConds(t0=0., tf=2e-7, N_t=20001,
                           x0=0.054094482, y0=0.03873037, z0=5.988900879,
                           p0=104.96, theta0=np.pi-np.pi/3, phi0=0.)
bounds_Mu2e = Bounds(xmin=-0.95, xmax=0.95, ymin=-0.95, ymax=0.95, zmin=3.239, zmax=14.139)
