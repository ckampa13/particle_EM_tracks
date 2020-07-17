import os
import numpy as np
import pandas as pd
import lmfit as lm
import plotly.io as pio

from hallprobecalib.hpcplots import histo
import matplotlib.pyplot as plt
from emtracks.plotting import config_plots

config_plots()

datadir = '/home/ckampa/data/pickles/distortions/linear_gradient/'
plotdir = '/home/ckampa/data/plots/distortions/linear_gradient/'

for d in [plotdir, plotdir+'run_04/', plotdir+'run_04/LHelix_reco/']:
    if not os.path.exists(d):
        os.makedirs(d)

### Mau10
# nice_name = 'Mau10 Combined Maps'
# dist_name = 'Mau10_0TS'
# file_suffix = 'Mau10_TSOff'
###
### Mau13
nice_name = 'Mau13 Subtracted Maps'
dist_name = 'Mau13_Subtractions'
file_suffix = 'Mau13_TSOff'
###

scales_dis = np.linspace(0,0.9,10)
scales = np.linspace(0,1,11)

df_run = pd.read_pickle(datadir+f'run_04/MC_sample_plus_reco_{file_suffix}.pkl')

df_run.loc[:, 'Residual Nominal'] = df_run['mom_nom'] - df_run['p_mc']
for s in scales_dis:
    df_run.loc[:, f'Residual Distorted {s:0.2f}xTS'] = df_run[f'mom_dis_{s:0.2f}TS'] - df_run['p_mc']

N = 75
N2 = 200

plot_file_pre = plotdir+f'run_04/LHelix_reco/{dist_name}/'

fig = histo([df_run['Residual Nominal']], bins=N, inline=False, show_plot=False)
fig.update_layout(xaxis=dict(title=r"$\text{Residual }(p_{\text{reco}} - p_{\text{MC}}) [\text{MeV } c^{-1}]$"),title=nice_name+' (nominal)')
pio.write_image(fig, plot_file_pre+'hist_residuals_nom.pdf')
pio.write_image(fig, plot_file_pre+'hist_residuals_nom.png')

for s in scales_dis:
    fig = histo([df_run[f'Residual Distorted {s:0.2f}xTS']], bins=N, inline=False, show_plot=False)
    fig.update_layout(xaxis=dict(title=r"$\text{Residual }(p_{\text{reco}} - p_{\text{MC}}) [\text{MeV } c^{-1}]$"),title=nice_name+f" (Distorted: {s:0.2f}x(PS+TS))")
    pio.write_image(fig, plot_file_pre+f'hist_residuals_dis_{s:0.2f}TS.pdf')
    pio.write_image(fig, plot_file_pre+f'hist_residuals_dis_{s:0.2f}TS.png')

fig = histo([df_run['Residual Nominal']]+[df_run[f'Residual Distorted {s:0.2f}xTS'] for s in scales_dis], bins=N2, same_bins=True, inline=False, show_plot=False)
fig.update_layout(xaxis=dict(title=r"$\text{Residual }(p_{\text{reco}} - p_{\text{MC}}) [\text{MeV } c^{-1}]$"),title=f"Distorted ({dist_name}: {s:0.2f}x(PS+TS))")
pio.write_image(fig, plot_file_pre+f'residual_mean_comparison.pdf')
pio.write_image(fig, plot_file_pre+f'residual_mean_comparison.png')

# linear fit
def lin(x, m, b):
    return m*x + b

def fit_linear(scales, means, stds):
    model = lm.Model(lin, independent_vars=['x'])
    params = lm.Parameters()
    m_guess = (means[-1]-means[0])/(scales[-1]-scales[0])
    b_guess = means[0]
    params.add('m', value=m_guess)
    params.add('b', value=b_guess)
    result = model.fit(means, x=scales, params=params, weights=1/stds)
    means_fit = result.eval(x=scales)
    m = result.params['m'].value
    b = result.params['b'].value
    chi2red = result.redchi
    return means_fit, m, b, chi2red

# new scatter plot showing everything
means = np.array([df_run[f'mom_dis_{i:0.2f}TS'].mean() for i in scales_dis]+[df_run['mom_nom'].mean()])
stds = np.array([df_run[f'mom_dis_{i:0.2f}TS'].std() for i in scales_dis]+[df_run['mom_nom'].std()])
means_fit, m, b, chi2red = fit_linear(scales, means, stds)
p_mc = df_run.p_mc.values[0]
# mean vs scale
fig = plt.figure()
plt.errorbar(scales, means, yerr=stds, fmt='.', markersize=10, capsize=6, capthick=2, elinewidth=2, label='Mean (point), StdDev. (error bar)')
plt.plot([0,1], [p_mc, p_mc], 'r--', label=f'True Momentum = {p_mc:0.3f} MeV/c')
plt.plot(scales, means_fit, '-.', c='gray', label=f'\nFit: mom = {m:0.3f} * scale + {b:0.3f}\n'+r'$\chi^2_{\mathrm{red.}}=$'+f'{chi2red:0.4f}\n')
plt.legend()
plt.xticks(scales)
plt.xlabel('PS+TS Scale')
plt.ylabel('Mean Reconstructed Momentum [MeV/c]')
plt.title(nice_name)
fig.tight_layout()
fig.savefig(plot_file_pre+f'mean_reco_mom_vs_TS_scale_{file_suffix}.pdf')
fig.savefig(plot_file_pre+f'mean_reco_mom_vs_TS_scale_{file_suffix}.png')

# std vs scale
fig = plt.figure()
plt.scatter(scales, stds, s=35, zorder=110, label='StDev. (point)')
if file_suffix == 'Mau13_TSOff':
    stds_fit, m_s, b_s, chi2red_s = fit_linear(scales, stds, np.ones_like(scales))
    plt.plot(scales, stds_fit, '-.', c='gray', zorder=100, label=f'\nFit: stddev = {m_s:0.3f} * scale + {b_s:0.3f}\n'+r'$\chi^2_{\mathrm{red.}}=$'+f'{chi2red_s:0.4f}\n')
    plt.legend()
plt.xticks(scales)
plt.xlabel('PS+TS Scale')
plt.ylabel('StdDev. Reconstructed Momentum [MeV/c]')
plt.title(nice_name)
fig.tight_layout()
fig.savefig(plot_file_pre+f'stddev_reco_mom_vs_TS_scale_{file_suffix}.pdf')
fig.savefig(plot_file_pre+f'stddev_reco_mom_vs_TS_scale_{file_suffix}.png')

