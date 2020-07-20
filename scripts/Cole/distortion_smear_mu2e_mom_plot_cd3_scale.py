import os
import numpy as np
import pandas as pd
import plotly.io as pio
import matplotlib.pyplot as plt

from hallprobecalib.hpcplots import histo
from emtracks.generate_momentum import *
from emtracks.plotting import config_plots

config_plots()

datadir = '/home/ckampa/data/pickles/distortions/linear_gradient/'
plotdir = '/home/ckampa/data/plots/distortions/linear_gradient/'

### Mau10
# nice_name = 'Mau10 Combined Maps'
# dist_name = 'Mau10_0TS'
# file_suffix = 'Mau10_TSOff'
###
### Mau13
nice_name = 'Mau13 Subtracted Maps'
dist_name = 'Mau13_Subtractions'
# file_suffix = 'Mau13_TSOff'
file_suffix = 'Mau13_DSOff'
###
if file_suffix == 'Mau13_TSOff':
    xtitle = 'PS+TS Scale'
    sol = 'TS'
else:
    xtitle = 'DS Scale'
    sol = 'DS'


# scales
# scales_coarse = np.linspace(0., 0.9, 10) # first coarse fields
scales_coarse = np.linspace(0.1, 0.8, 8) # scale DS, first coarse fields
scales_fine = np.concatenate([np.linspace(.91, .99, 9), np.linspace(1.01, 1.10,10)])# new fields
scales_dis = np.concatenate([scales_coarse, scales_fine])# course fields + new fields
scales = sorted(np.concatenate([scales_dis, np.array([1.0])]))

# scales_dis = np.linspace(0,0.9,10)
# scales = np.linspace(0,1,11)

# significance function
def signal_significance(s, b):
    # S = sqrt(2 * ((s+b) * ln(1+s/b) -s))
    S = (2*((s+b)*np.log(1+s/b)-s))**(1/2)
    return S

# scale for event generation
scale = 10000
# plot and cut windows
p_low_plot = 102.
p_hi_plot = 106.
p_low_cut = 103.85
p_hi_cut = 104.90

# load emtracks p residuals
df_run = pd.read_pickle(datadir+f'run_04/MC_sample_plus_reco_{file_suffix}.pkl')

df_run.loc[:, 'Residual Nominal'] = df_run['mom_nom'] - df_run['p_mc']
for s in scales_dis:
    df_run.loc[:, f'Residual Distorted {s:0.2f}x{sol}'] = df_run[f'mom_dis_{s:0.2f}{sol}'] - df_run['p_mc']

plot_file_pre = plotdir+f'run_04/LHelix_reco/{dist_name}/{sol}_scale/'

# load digitization file
digit_file = '/home/ckampa/data/root/cd3_ce_and_background_digitized.csv'
df_dig = pd.read_csv(digit_file, names=["mom_DIO", "N_DIO", "mom_CE", "N_CE"], skiprows=2)
mom_CE, N_CE_dig = df_dig[["mom_CE", "N_CE"]].values.T
mom_DIO, N_DIO_dig = df_dig[["mom_DIO","N_DIO"]].dropna().values.T
# adjust for all other backgrounds
N_DIO_dig = N_DIO_dig - N_DIO_dig[-1]
cut_CE = (mom_CE >= p_low_cut) & (mom_CE <= p_hi_cut)
cut_DIO = (mom_DIO >= p_low_cut) & (mom_DIO <= p_hi_cut)
N_CE_window_dig = N_CE_dig[cut_CE].sum()
N_DIO_window_dig = N_DIO_dig[cut_DIO].sum()

# generate scaled number of CE and DIO events
p_ce = generate_mu2e_event(N=N_CE_CD3*scale, interaction="ce", low=p_low_plot, high=p_hi_plot)
p_dio = generate_mu2e_event(N=N_DIO_CD3*scale, interaction="dio", low=p_low_plot, high=E_endpoint)
# counts for nominal
N_CE_window = ((p_ce >= p_low_cut) & (p_ce <= p_hi_cut)).sum() / scale
N_DIO_window = ((p_dio >= p_low_cut) & (p_dio <= p_hi_cut)).sum() / scale
# plot final result
label_temp = '{0}\n' + r'$\mu = {1:.3E}$'+ '\n' + 'std' + r'$= {2:.3E}$' + '\n' +  'Integral: {3:.1f}\n' 'N in momentum window: {4:.2f}\n' + 'N (CD3): {5:.2f}\n'
bins = np.linspace(102., 106., 81)
# original results
fig, ax = plt.subplots()
ax.hist(p_ce, edgecolor='red', bins=bins, weights=1/scale*np.ones_like(p_ce), histtype='step', linewidth=2.,
        label=label_temp.format('Signal e-:', np.mean(p_ce), np.std(p_ce), len(p_ce)/scale, N_CE_window, N_CE_window_dig)
        +f'Signal Significance: {signal_significance(N_CE_window, N_DIO_window):0.1f}\n')
ax.hist(p_dio, edgecolor='blue', bins=bins, weights=1/scale*np.ones_like(p_dio), histtype='step', linewidth=2.,
        label=label_temp.format('DIO e-:', np.mean(p_dio), np.std(p_dio), len(p_dio)/scale, N_DIO_window, N_DIO_window_dig))
ax.plot([p_low_cut, p_low_cut], [0, 0.45], c='gray', linestyle='--', label=f'Momentum Window:\n[{p_low_cut:.2f}, {p_hi_cut:.2f}] MeV/c\n')
ax.plot([p_hi_cut, p_hi_cut], [0, 0.45], c='gray', linestyle='--')
ax.set_xlabel('Track momentum, MeV/c')
ax.set_ylabel('Events per 0.05 MeV/c')
ax.set_title('CD3 Era Parameterization\nOriginal Results')
ax.set_ylim([0,1.])
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig(plot_file_pre+'cd3_ce_dio_plot_original.pdf')
fig.savefig(plot_file_pre+'cd3_ce_dio_plot_original.png')


def make_dist_plot(sol_scale=0.0):
    scalestr = f'{sol_scale:0.2f}'
    # generate errors
    mom_errs_ce = df_run[f"Residual Distorted {scalestr}x{sol}"].sample(len(p_ce), replace=True).values
    mom_errs_dio = df_run[f"Residual Distorted {scalestr}x{sol}"].sample(len(p_dio), replace=True).values
    p_ce_adj = p_ce + mom_errs_ce
    p_dio_adj = p_dio + mom_errs_dio
    # count number in signal window
    N_CE_window_adj = ((p_ce_adj >= p_low_cut) & (p_ce_adj <= p_hi_cut)).sum() / scale
    N_DIO_window_adj = ((p_dio_adj >= p_low_cut) & (p_dio_adj <= p_hi_cut)).sum() / scale
    # smeared results plot
    fig2, ax2 = plt.subplots()
    ax2.hist(p_ce_adj, edgecolor='red', bins=bins, weights=1/scale*np.ones_like(p_ce_adj), histtype='step', linewidth=2.,
             label=label_temp.format('Signal e-:', np.mean(p_ce_adj), np.std(p_ce_adj), len(p_ce_adj)/scale, N_CE_window_adj, N_CE_window_dig)
             +f'Signal Significance: {signal_significance(N_CE_window_adj, N_DIO_window_adj):0.1f}\n')
    ax2.hist(p_dio_adj, edgecolor='blue', bins=bins, weights=1/scale*np.ones_like(p_dio_adj), histtype='step', linewidth=2.,
            label=label_temp.format('DIO e-:', np.mean(p_dio_adj), np.std(p_dio_adj), len(p_dio_adj)/scale, N_DIO_window_adj, N_DIO_window_dig))
    ax2.plot([p_low_cut, p_low_cut], [0, 0.45], c='gray', linestyle='--', label=f'Momentum Window:\n[{p_low_cut:.2f}, {p_hi_cut:.2f}] MeV/c\n')
    ax2.plot([p_hi_cut, p_hi_cut], [0, 0.45], c='gray', linestyle='--')
    ax2.set_xlabel('Track momentum, MeV/c')
    ax2.set_ylabel('Events per 0.05 MeV/c')
    ax2.set_title(f'CD3 Era Parameterization\nWith Sampled B Distortion Momentum Errors ({nice_name}: {scalestr}x({sol}))')
    ax2.set_ylim([0,1.])
    ax2.legend(loc='upper right')
    fig2.tight_layout()
    fig2.savefig(plot_file_pre+f'cd3_ce_dio_plot_smeared_{scalestr}{sol}.pdf')
    fig2.savefig(plot_file_pre+f'cd3_ce_dio_plot_smeared_{scalestr}{sol}.png')

# make distorted plots
for s in scales_dis:
    make_dist_plot(s)
