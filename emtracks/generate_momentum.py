import math
import numpy as np
import pandas as pd
import scipy.optimize as optimization
from scipy import stats

import pypdt

# from digitization + normalization check
N_DIO_CD3 = 65 # 72
N_CE_CD3 = 10
p_range_CD3 = [102., 106.]

# e mass
m_e = 1000. * pypdt.get(11).mass

# DIO parameterization
a5, a6, a7, a8 = (8.6434e-17, 1.16874e-17, -1.87828e-19, 9.16327e-20)
E_mu = 105.194
E_endpoint = 104.973
m_al = 25133

def dio(mom_e):
    E_e = (mom_e**2 + m_e**2)**(1/2)
    # if E_e > E_endpoint:
    #     return 0.
    delta = E_mu - E_e - np.square(E_e) / (2 * m_al)
    f = a5 * delta**5 + a6 * delta**6 + a7 * delta**7 + a8 * delta**8
    if type(mom_e) is float:
        if E_e > E_endpoint:
            f = 0.
    else:
        f[E_e > E_endpoint] = 0.
    return f

# CE parameterization
CE_file = '/home/ckampa/data/root/ce_mom.csv'
mom = pd.read_csv(CE_file, names=['index', 'mom'])
mom = mom.mom
mom = mom[(mom > 98.) & (mom<110.)] # picks out right range for Crystal Ball fit
# run kde to use for fitter
positions = np.linspace(100., 104.9, 200)
kernel = stats.gaussian_kde(mom)
mom_kde = kernel(positions)

def crystal_ball(x, beta, m, loc, scale):
    return stats.crystalball.pdf(x, beta, m, loc, scale)

guess = np.array([1, 2, 104., 1.]) # from previous work (stats project)
x_dat = positions
params, cov = optimization.curve_fit(crystal_ball, x_dat, mom_kde, guess)
beta, m, loc, scale = params
y_max_ce = crystal_ball(loc, beta, m, loc, scale)

def ce(mom):
    return stats.crystalball.pdf(mom, beta, m, loc, scale)

# generator function
def generate_mu2e_event(N, interaction, low, high):
    # N: number of events to generate,
    # interaction: {"dio", "ce"}
    # efficiency: estimated sampling efficiency
    if interaction == "dio":
        ymax = dio(low)
        pdf_func = dio
        efficiency = 0.16
    elif interaction == "ce":
        ymax = y_max_ce
        pdf_func = ce
        efficiency = 0.6
    else:
        raise NotImplementedError('interaction is not in the list of supported interactions: ["ce", "dio"]')
    mom = np.array([])
    Ngen = math.ceil(N/efficiency) # account for sampling efficiency
    while len(mom) < N:
        xs = np.random.uniform(low=low, high=high, size=Ngen)
        ys = np.random.uniform(low=0, high=ymax, size=Ngen)
        passed = ys <= pdf_func(xs)
        mom = np.concatenate([mom, xs[passed]])
    return mom[:N]
