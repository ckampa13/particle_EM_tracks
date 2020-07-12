import os
import numpy as np
import pandas as pd
import plotly.io as pio
import matplotlib.pyplot as plt

from hallprobecalib.hpcplots import histo
from emtracks.plotting import config_plots

config_plots()

datadir = '/home/ckampa/data/pickles/distortions/linear_gradient/'
plotdir = '/home/ckampa/data/plots/distortions/linear_gradient/'
# ntupfile = '/home/ckampa/data/root/ce_OLD.pkl'
ntupfile = '/home/ckampa/data/root/ensemble_01_trkana_full_00.pkl'

# load emtracks p residuals
df_run = pd.read_pickle(datadir+'run_04/MC_sample_plus_reco_chi2.pkl')
df_run.loc[:, 'Residual Nominal'] = df_run['p_chi2_nom'] - df_run['p_mc']
df_run.loc[:, 'Residual Distorted'] = df_run['p_chi2_dis'] - df_run['p_mc']

# load ntuple and filter
df = pd.read_pickle(ntupfile)
# # cut set 'C'
# kf_mask = df['de.status'] > 0 # Kalman fit status
# na_mask = df['de.nactive'] >= 25 # number of track hits in fit
# con_mask = df['de.fitcon'] > 2e-3 # fit consistency
# merr_mask = df['deent.momerr'] < 0.25 # estimated momentum error
# terr_mask = df['de.t0err'] < 0.9 # estimated reco time at tracker center (z=0)
# d0_mask = (df['deent.d0'] > -80.) & (df['deent.d0'] < 105.) # consistent w coming from stopping target
# tr_mask = (df['deent.d0'] + 2 / df['deent.om'] > 450.) & (df['deent.d0'] + 2 / df['deent.om'] < 680.) # inconsistent w proton absorber
# t0_mask = (df['de.t0'] > 700.) & (df['de.t0'] < 1695.) # set by RPC
# td_mask = (df['deent.td'] > 0.57735) & (df['deent.td'] < 1.) # exclude beam particles

# cutset_C = kf_mask & na_mask & con_mask & merr_mask & terr_mask & d0_mask & tr_mask & t0_mask & td_mask
# df = df[cutset_C].copy()
# df.reset_index(drop=True, inplace=True)

# define crv variable to cut on
delta = []
for row in df.iterrows():
    if len(row[1]["crvinfo._timeWindowStart"]) > 0:
        delta.append(row[1]['crvinfo._timeWindowStart'][0] - row[1]['de.t0'])
    else:
        delta.append(-10000)
df['crv_delta_t0'] = pd.Series(delta)

# ensemble cuts
t0_mask = (df['de.t0'] > 700.) & (df['de.t0'] < 1695.) # set by RPC
trkqual_mask = df['dequal.TrkQual'] > 0.8
PID_mask = df['dequal.TrkPID'] > 0.95
td_mask = (df['deent.td'] > 0.57735) & (df['deent.td'] < 1.) # exclude beam particles
d0_mask = (df['deent.d0'] > -80.) & (df['deent.d0'] < 105.) # consistent w coming from stopping target
tr_mask = (df['deent.d0'] + 2 / df['deent.om'] > 450.) & (df['deent.d0'] + 2 / df['deent.om'] < 680.) # inconsistent w proton absorber
crv_mask = ((df['crv_delta_t0'] < -120) | (df['crv_delta_t0'] > 50)) # no crv trig in certain time window
up_mask = df['tcnt.nue'] == 0 # no upstream tracks
trig_mask = df['tcnt.nde'] > 0 # at least one downstream e- track

cutset_ens = t0_mask & trkqual_mask & PID_mask & td_mask & d0_mask & tr_mask & crv_mask & up_mask & trig_mask
df = df[cutset_ens].copy()

p_low_plot = 95.
p_hi_plot = 115.
# mom_window = (df['deent.mom'] >= p_low_plot) & (df['deent.mom'] <= p_hi_plot)
# df = df[mom_window].copy()
df.reset_index(drop=True, inplace=True)
# sample momentum errors
mom_errs = df_run["Residual Distorted"].sample(len(df), replace=True).values
df['mom_adjusted'] = df['deent.mom'] + mom_errs
# pick out momentum values
df_ce = df[df["demc.gen"] == 43].copy()
df_ce.reset_index(drop=True, inplace=True)
df_dio = df[df["demc.gen"] == 7].copy()
df_dio.reset_index(drop=True, inplace=True)
mom_ce = (df_ce['deent.mom'] >= p_low_plot) & (df_ce['deent.mom'] <= p_hi_plot)
mom_ce_adj = (df_ce['mom_adjusted'] >= p_low_plot) & (df_ce['mom_adjusted'] <= p_hi_plot)
mom_dio = (df_dio['deent.mom'] >= p_low_plot) & (df_dio['deent.mom'] <= p_hi_plot)
mom_dio_adj = (df_dio['mom_adjusted'] >= p_low_plot) & (df_dio['mom_adjusted'] <= p_hi_plot)
p_ce = df_ce[mom_ce]['deent.mom'].values
p_dio = df_dio[mom_dio]['deent.mom'].values
p_ce_adj = df_ce[mom_ce_adj]['mom_adjusted'].values
p_dio_adj = df_dio[mom_dio_adj]['mom_adjusted'].values
# # sample momentum errors
# dio_errs = df_run["Residual Distorted"].sample(len(df_dio), replace=True).values
# ce_errs = df_run["Residual Distorted"].sample(len(df_ce), replace=True).values
# # smear reco momenta
# df_dio['mom_adjusted'] = df_dio['deent.mom'] + dio_errs
# df_ce['mom_adjusted'] = df_ce['deent.mom'] + ce_errs
# count number in momentum window
p_low = 103.85
p_hi = 104.90
N_ce = ((df_ce['deent.mom'] >= 103.85) & (df_ce['deent.mom'] <= 104.9)).sum()
N_dio = ((df_dio['deent.mom'] >= 103.85) & (df_dio['deent.mom'] <= 104.9)).sum()
N_ce_adj = ((df_ce['mom_adjusted'] >= 103.85) & (df_ce['mom_adjusted'] <= 104.9)).sum()
N_dio_adj = ((df_dio['mom_adjusted'] >= 103.85) & (df_dio['mom_adjusted'] <= 104.9)).sum()

# plot final result
label_temp = '{0}\n' + r'$\mu = {1:.3E}$'+ '\n' + 'std' + r'$= {2:.3E}$' + '\n' +  'Integral: {3}\n' 'N in momentum window: {4}\n'
# bins = np.linspace(102., 106., 81)
bins = np.linspace(95., 115., 41)
# original results
fig, ax = plt.subplots()
ax.hist(df_ce['deent.mom'], bins=bins, histtype='step', label=label_temp.format('Signal e-:', np.mean(p_ce), np.std(p_ce), len(p_ce), N_ce))
ax.hist(df_dio['deent.mom'], bins=bins, histtype='step', label=label_temp.format('DIO e-:', np.mean(p_dio), np.std(p_dio), len(p_dio), N_dio))
ax.plot([p_low, p_low], [0, 120], 'r--', label=f'Momentum Window:\n[{p_low:.2f}, {p_hi:.2f}] MeV/c\n')
ax.plot([p_hi, p_hi], [0, 120], 'r--')
ax.set_xlabel('Track momentum, MeV/c')
# ax.set_ylabel('Events per 0.05 MeV/c')
ax.set_ylabel('Events per 0.5 MeV/c')
ax.set_title('MDC 2018 Ensemble 01\nOriginal Results')
ax.set_ylim([0,105])
# fig.legend(loc='lower right')
ax.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
fig.tight_layout()
fig.savefig(plotdir+'ce_dio_plot_original.pdf')
fig.savefig(plotdir+'ce_dio_plot_original.png')

# smeared results
fig, ax = plt.subplots()
ax.hist(df_ce['mom_adjusted'], bins=bins, histtype='step', label=label_temp.format('Signal e-:', np.mean(p_ce_adj), np.std(p_ce_adj), len(p_ce_adj), N_ce_adj))
ax.hist(df_dio['mom_adjusted'], bins=bins, histtype='step', label=label_temp.format('DIO e-:', np.mean(p_dio_adj), np.std(p_dio_adj), len(p_dio_adj), N_dio_adj))
ax.plot([p_low, p_low], [0, 120], 'r--', label=f'Momentum Window:\n[{p_low:.2f}, {p_hi:.2f}] MeV/c\n')
ax.plot([p_hi, p_hi], [0, 120], 'r--')
ax.set_xlabel('Track momentum, MeV/c')
# ax.set_ylabel('Events per 0.05 MeV/c')
ax.set_ylabel('Events per 0.5 MeV/c')
ax.set_title('MDC 2018 Ensemble 01\nWith B Linear Gradient Momentum Errors')
ax.set_ylim([0,105])
# fig.legend(loc='lower right')
ax.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
fig.tight_layout()
fig.savefig(plotdir+'ce_dio_plot_smeared.pdf')
fig.savefig(plotdir+'ce_dio_plot_smeared.png')
