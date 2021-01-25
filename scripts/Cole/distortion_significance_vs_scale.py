import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from emtracks.generate_momentum import *
from emtracks.plotting import config_plots

config_plots()

datadir = '/home/ckampa/data/pickles/distortions/linear_gradient/'
plotdir = '/home/ckampa/data/plots/distortions/linear_gradient/'

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
scales_coarse = np.linspace(0., 0.9, 10) # first coarse fields
# scales_coarse = np.linspace(0.1, 0.8, 8) # scale DS, first coarse fields
scales_fine = np.concatenate([np.linspace(.91, .99, 9), np.linspace(1.01, 1.10,10)])# new fields
if file_suffix == 'Mau13_TSOff':
    scales_dis = np.concatenate([scales_coarse, scales_fine])# course fields + new fields
elif file_suffix == 'Mau13_DSOff':
    scales_dis = np.concatenate([[0.98, 0.99], np.linspace(.995, .999, 5), np.linspace(1.001, 1.005, 5), [1.01, 1.02]])
scales_dis_str = [f'{scale:.2f}' if int(round(scale*1000 % 10)) == 0 else f'{scale:.3f}' for scale in scales_dis]
scales = sorted(np.concatenate([scales_dis, np.array([1.0])]))
# scales = np.linspace(0.995, 1.005, 11)
scales_str = [f'{scale:.2f}' if int(round(scale*1000 % 10)) == 0 else f'{scale:.3f}' for scale in scales]

# scales_dis = np.linspace(0,0.9,10)
# scales = np.linspace(0,1,11)

# significance function
def signal_significance(s, b):
    # S = sqrt(2 * ((s+b) * ln(1+s/b) -s))
    if b == 0:
        # return -1
        b = 1e-10
    S = (2*((s+b)*np.log(1+s/b)-s))**(1/2)
    return S

# scale for event generation
scale = 10000
# plot and cut windows
p_low_plot = 102.
p_hi_plot = 106.
p_low_cut = 103.85
p_hi_cut = 104.90
window_width = p_hi_cut - p_low_cut
bin_width=0.05
N_OTHER = 0.01415 * window_width/bin_width # from plot digitization
MeVc_per_Scale = 104.6452 # from mean momentum vs DS scale

# load emtracks p residuals
df_run = pd.read_pickle(datadir+f'run_04/MC_sample_plus_reco_{file_suffix}.pkl')

df_run.loc[:, 'Residual Nominal'] = df_run['mom_nom'] - df_run['p_mc']
for s in scales_dis_str:
    df_run.loc[:, f'Residual Distorted {s}x{sol}'] = df_run[f'mom_dis_{s}{sol}'] - df_run['p_mc']

plot_file_pre = plotdir+f'run_04/LHelix_reco/{dist_name}/{sol}_scale/'

# load digitization file
digit_file = '/home/ckampa/data/root/cd3_ce_and_background_digitized.csv'
df_dig = pd.read_csv(digit_file, names=["mom_DIO", "N_DIO", "mom_CE", "N_CE"], skiprows=2)
mom_CE, N_CE_dig = df_dig[["mom_CE", "N_CE"]].values.T
mom_DIO, N_DIO_dig = df_dig[["mom_DIO","N_DIO"]].dropna().values.T
dioargs = mom_DIO.argsort()
ceargs = mom_CE.argsort()
mom_CE = mom_CE[ceargs]
N_CE_dig = N_CE_dig[ceargs]
mom_DIO = mom_DIO[dioargs]
N_DIO_dig = N_DIO_dig[dioargs]

# generate scaled number of CE and DIO events
p_ce = generate_mu2e_event(N=N_CE_CD3*scale, interaction="ce", low=p_low_plot, high=p_hi_plot)
p_dio = generate_mu2e_event(N=N_DIO_CD3*scale, interaction="dio", low=p_low_plot, high=E_endpoint)


def gen_N_CE_N_DIO(scale_str = '0.995', nominal=False):
    # generate errors
    if nominal:
        mom_errs_ce = df_run[f"Residual Nominal"].sample(len(p_ce), replace=True).values
        mom_errs_dio = df_run[f"Residual Nominal"].sample(len(p_dio), replace=True).values
    else:
        mom_errs_ce = df_run[f"Residual Distorted {scale_str}x{sol}"].sample(len(p_ce), replace=True).values
        mom_errs_dio = df_run[f"Residual Distorted {scale_str}x{sol}"].sample(len(p_dio), replace=True).values
    p_ce_adj = p_ce + mom_errs_ce
    p_dio_adj = p_dio + mom_errs_dio
    # count number in signal window
    N_CE = ((p_ce_adj >= p_low_cut) & (p_ce_adj <= p_hi_cut)).sum() / scale
    N_DIO = ((p_dio_adj >= p_low_cut) & (p_dio_adj <= p_hi_cut)).sum() / scale
    return N_CE, N_DIO

def N_window_vs_scale(scale=.995):
    delta_p = MeVc_per_Scale * (scale-1.0)
    p_lo = p_low_cut + delta_p
    p_hi = p_hi_cut + delta_p
    mask_DIO = (mom_DIO>=p_lo) & (mom_DIO<=p_hi)
    mask_CE = (mom_CE>=p_lo) & (mom_CE<=p_hi)
    N0_DIO = N_DIO_dig[mask_DIO][0]*(mom_DIO[mask_DIO][0] - p_lo)/bin_width
    Nf_DIO = N_DIO_dig[mask_DIO][-1]*(p_hi - mom_DIO[mask_DIO][-1])/bin_width
    N0_CE = N_CE_dig[mask_CE][0]*(mom_CE[mask_CE][0] - p_lo)/bin_width
    Nf_CE = N_CE_dig[mask_CE][-1]*(p_hi - mom_CE[mask_CE][-1])/bin_width
    N_DIO_window = N_DIO_dig[mask_DIO][1:-1].sum() + N0_DIO + Nf_DIO
    N_CE_window = N_CE_dig[mask_CE][1:-1].sum() + N0_CE + Nf_CE
    return N_CE_window, N_DIO_window

# from reco / scaling
N_CEs = []
N_DIOs = []
sigs = []
for s in scales_str:
    if s == '1.00':
        N_CE, N_DIO = gen_N_CE_N_DIO(nominal=True)
    else:
        N_CE, N_DIO = gen_N_CE_N_DIO(scale_str=s)
    sig = signal_significance(N_CE, N_DIO + N_OTHER)
    N_CEs.append(N_CE)
    N_DIOs.append(N_DIO)
    sigs.append(sig)
N_CEs = np.array(N_CEs)
N_DIOs = np.array(N_DIOs)
sigs = np.array(sigs)

# from digitization
# scales_dig = np.linspace(0.99,1.01, 21)
scales_dig = np.linspace(0.993,1.02, 100)
N_CE_digs = []
N_DIO_digs = []
sig_digs = []
for s in scales_dig:
    N_CE, N_DIO = N_window_vs_scale(scale=s)
    sig = signal_significance(N_CE, N_DIO)
    N_CE_digs.append(N_CE)
    N_DIO_digs.append(N_DIO)
    sig_digs.append(sig)
N_CE_digs = np.array(N_CE_digs)
N_DIO_digs = np.array(N_DIO_digs)
sig_digs = np.array(sig_digs)

# plot significance
fig = plt.figure()
plt.scatter(scales, sigs, s=25)
plt.xlabel(f'{sol} Scale')
plt.ylabel('Signal Significance')
fig.savefig(plot_file_pre+f'significance_vs_{sol}_scale.pdf')
fig.savefig(plot_file_pre+f'significance_vs_{sol}_scale.png')

# plot Ns
fig = plt.figure()
plt.scatter(scales, N_CEs, s=25, label='Conversion e-')
plt.scatter(scales, N_DIOs, s=25, label='DIO e-')
plt.xlabel(f'{sol} Scale')
plt.ylabel('Number in Signal Window')
plt.legend()
fig.savefig(plot_file_pre+f'N_compare_vs_{sol}_scale.pdf')
fig.savefig(plot_file_pre+f'N_compare_vs_{sol}_scale.png')

# plot together
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel(f'{sol} Scale')
ax1.set_ylabel(f'Number in Signal Window', color=color)
ax1.scatter(scales, N_CEs, c=color, marker='.', s=50, label='Conversion e-')
ax1.scatter(scales, N_DIOs+N_OTHER, c=color, marker='+', s=50, label='Backgrounds')
ax1.tick_params(axis='y', labelcolor=color)
# ax1.legend()

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Signal Significance', color=color)
# if file_suffix == 'Mau13_DSOff':
#     ax2.plot(scales_dig, sig_digs, '-', c=color, linewidth=1, label='Significance (digitized CD3 Plot)')
ax2.scatter(scales, sigs, c=color, marker='s', s=25, label='Significance')
ax2.tick_params(axis='y', labelcolor=color)
# ax2.legend()

fig.legend()

fig.savefig(plot_file_pre+f'significance-N_vs_{sol}_scale.pdf')
fig.savefig(plot_file_pre+f'significance-N_vs_{sol}_scale.png')
