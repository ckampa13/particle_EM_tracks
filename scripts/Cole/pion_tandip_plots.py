import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from emtracks.plotting import config_plots

config_plots()

ddir = "/home/ckampa/data/"
pkldir = ddir + 'pickles/emtracks/pion_degrader_tandip/'
plotdir = ddir + 'plots/emtracks/pion_degrader_tandip/'
# filenames
pklfile = pkldir+'degrader_tandip_df_run01.pkl'

# tracker cut
Rmin = 0. # 0.3769 # m
# bins
bins = np.linspace(0, 2, 101)
# bins = np.linspace(0, 2, 201)

# load results
df = pd.read_pickle(pklfile)
N_gen = len(df)

# a few functions to keep code DRY
def get_tand(df, name, Rmin=Rmin):
    tand = df.query(f'Rmax_{name} >= {Rmin}')[f'tand_{name}'].values
    return tand

# reference label
# label_temp = '{0}\n' + r'$\mu = {1:.3E}$'+ '\n' + 'std' + r'$= {2:.3E}$' + '\n' +  'Integral: {3}\n' + 'Underflow: {4}\nOverflow: {5}'

def get_label(name, tand, bins):
    over = (tand > np.max(bins)).sum()
    under = (tand < np.min(bins)).sum()
    # label = f'{name}\n' + rf'$\mu: {np.mean(tand):.3E}$' + '\n' + rf'$\sigma: {np.std(tand):.3E}$' + '\n' + f'Integral: {len(tand)}\nUnderflow: {under}\nOverflow: {over}'
    mean = f'{np.mean(tand):.3E}'
    std = f'{np.std(tand):.3E}'
    n = 15
    label = f'mean: {mean:>15}' + '\n' + f'stddev: {std:>15}' + '\n' + f'Integral: {len(tand):>17}\nUnderflow: {under:>16}\nOverflow: {over:>16}'
    return label

def make_plot(df, name, bins=bins):
    tand = get_tand(df, name)
    fig = plt.figure()
    plt.hist(tand, bins=bins, histtype='step', linewidth=1.5, label=get_label(name, tand, bins))
    plt.xlabel(r'$\tan(\mathrm{dip})$')
    plt.ylabel('Events')
    plt.title(f'{name}: 105 MeV Isotropic e+ Uniformly Generated in Pion Degrader\n'+r'$N_{\mathrm{gen}} = 1\times10^4$'+'; Tracker Cut ('+r'$R_{\mathrm{max}} >= $'+rf'${Rmin}$'+' [m])')
    plt.legend()
    fig.savefig(plotdir+f'{name}_degrader_tandip_hist.pdf')
    fig.savefig(plotdir+f'{name}_degrader_tandip_hist.png')

if __name__=='__main__':
    # names = [f'Mau{i}' for i in [9]]
    names = [f'Mau{i}' for i in [9, 10, 12, 13]]

    for name in names:
        make_plot(df, name)
