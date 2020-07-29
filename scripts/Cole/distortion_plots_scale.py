import os
import numpy as np
import pandas as pd
import lmfit as lm
import plotly.io as pio

from hallprobecalib.hpcplots import histo
import matplotlib.pyplot as plt
from emtracks.plotting import config_plots

config_plots()

# ACC_SCALE = 0.1 # MeV/c
ACC_SCALE = 0.04 # MeV/c

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
# scales_coarse = np.linspace(0., 0.8, 9) # scale TS, first coarse fields
scales_coarse = np.linspace(0.1, 0.8, 8) # scale DS, first coarse fields
scales_fine = np.concatenate([np.linspace(.9, .99, 10), np.linspace(1.01, 1.10,10)])# new fields
scales_fine_str = [f'{scale:.2f}' if int(round(scale*1000 % 10)) == 0 else f'{scale:.3f}' for scale in scales_fine]
# scales_finer = np.linspace(.995, 1.005, 11)
scales_finer = np.concatenate([np.linspace(.995, .999, 5), np.linspace(1.001, 1.005,5)])# new fields
scales_finer_str = [f'{scale:.2f}' if int(round(scale*1000 % 10)) == 0 else f'{scale:.3f}' for scale in scales_finer]
# scales_dis = np.concatenate([scales_coarse, scales_fine])# course fields + new fields
scales_dis = np.concatenate([scales_coarse, scales_fine, scales_finer])# course fields + new fields
scales_dis_str = [f'{scale:.2f}' if int(round(scale*1000 % 10)) == 0 else f'{scale:.3f}' for scale in scales_dis]
scales = np.concatenate([scales_dis, np.array([1.0])])
scales_ticks = np.linspace(0, 1.1, 12)
# scales_new = np.concatenate([scales_fine, np.array([1.0])])
# scales_new_ticks = np.linspace(0.9, 1.1, 21)
scales_new = np.concatenate([scales_finer, np.array([1.0])])
scales_new_ticks = np.linspace(0.995, 1.005, 11)

# scales_dis = np.linspace(0,0.9,10)
# scales = np.linspace(0,1,11)

df_run = pd.read_pickle(datadir+f'run_04/MC_sample_plus_reco_{file_suffix}.pkl')

df_run.loc[:, 'Residual Nominal'] = df_run['mom_nom'] - df_run['p_mc']
for s in scales_dis_str:
    df_run.loc[:, f'Residual Distorted {s}x{sol}'] = df_run[f'mom_dis_{s}{sol}'] - df_run['p_mc']

N = 75
N2 = 200

plot_file_pre = plotdir+f'run_04/LHelix_reco/{dist_name}/{sol}_scale/'

fig = histo([df_run['Residual Nominal']], bins=N, inline=False, show_plot=False)
fig.update_layout(xaxis=dict(title=r"$\text{Residual }(p_{\text{reco}} - p_{\text{MC}}) [\text{MeV } c^{-1}]$"),title=nice_name+' (nominal)')
pio.write_image(fig, plot_file_pre+'hist_residuals_nom.pdf')
pio.write_image(fig, plot_file_pre+'hist_residuals_nom.png')

for s in scales_dis_str:
    fig = histo([df_run[f'Residual Distorted {s}x{sol}']], bins=N, inline=False, show_plot=False)
    fig.update_layout(xaxis=dict(title=r"$\text{Residual }(p_{\text{reco}} - p_{\text{MC}}) [\text{MeV } c^{-1}]$"),title=nice_name+f" (Distorted: {s}x({sol}))")
    pio.write_image(fig, plot_file_pre+f'hist_residuals_dis_{s}{sol}.pdf')
    pio.write_image(fig, plot_file_pre+f'hist_residuals_dis_{s}{sol}.png')

fig = histo([df_run['Residual Nominal']]+[df_run[f'Residual Distorted {s}x{sol}'] for s in scales_dis_str], bins=N2, same_bins=True, inline=False, show_plot=False)
fig.update_layout(xaxis=dict(title=r"$\text{Residual }(p_{\text{reco}} - p_{\text{MC}}) [\text{MeV } c^{-1}]$"),title=f"Distorted ({dist_name})")
pio.write_image(fig, plot_file_pre+f'residual_mean_comparison_{sol}_scale.pdf')
pio.write_image(fig, plot_file_pre+f'residual_mean_comparison_{sol}_scale.png')


# linear fit
def lin(x, m, b):
    return m*x + b

def quad(x, a, b, c):
    return a*x**2 + b*x + c

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

def fit_quadratic(scales, means, stds):
    model = lm.Model(quad, independent_vars=['x'])
    params = lm.Parameters()
    c_guess = means[0]
    b_guess = 2*(means.min()-c_guess) / scales[means.argmin()]
    a_guess = -b_guess / (2*scales[means.argmin()])
    params.add('a', value=a_guess)
    params.add('b', value=b_guess)
    params.add('c', value=c_guess)
    result = model.fit(means, x=scales, params=params, weights=1/stds)
    means_fit = result.eval(x=scales)
    a = result.params['a'].value
    b = result.params['b'].value
    c = result.params['c'].value
    chi2red = result.redchi
    return means_fit, a, b, c, chi2red

def plot_mean_scatter(scales_dis_str, scales, scales_ticks, file_suffix2=''):
    # new scatter plot showing everything
    means = np.array([df_run[f'mom_dis_{i}{sol}'].mean() for i in scales_dis_str]+[df_run['mom_nom'].mean()])
    stds = np.array([df_run[f'mom_dis_{i}{sol}'].std() for i in scales_dis_str]+[df_run['mom_nom'].std()])
    means_fit, m, b, chi2red = fit_linear(scales, means, stds)
    scales_fit = np.copy(scales)
    p_mc = df_run.p_mc.values[0]
    scale_center = (p_mc - b) / m
    delta_scale = ACC_SCALE / m
    if scale_center + delta_scale > scales.max():
        scales_fit = np.append(scales_fit, scale_center+delta_scale)
        means_fit = np.append(means_fit, m*(scale_center+delta_scale)+b)
    # mean vs scale
    fig = plt.figure()
    plt.errorbar(scales, means, yerr=stds, fmt='.', markersize=10, capsize=6, capthick=2, elinewidth=2, zorder=53, label='Mean (point), StdDev. (error bar)')
    plt.plot([scales_fit.min(),scales_fit.max()], [p_mc, p_mc], 'r--', zorder=51, label=f'True Momentum = {p_mc:0.3f} MeV/c')
    plt.plot(scales_fit, means_fit, '-.', c='gray', zorder=52, label=f'\nFit: mom = {m:0.4f} * scale + {b:0.4f}\n'+r'$\chi^2_{\mathrm{red.}}=$'+f'{chi2red:0.4f}\n')
    # plt.fill_between([scales.min(), scales.max()], p_mc-ACC_SCALE, p_mc + ACC_SCALE, color='blue', alpha=0.2, zorder=49, label=r'Allowed $p_{\mathrm{reco.}}$ Scale: [$p_{\mathrm{true}}-$'+f'{ACC_SCALE}, '+r'$p_{\mathrm{true}}+$'+f'{ACC_SCALE}] MeV/c')
    plt.fill_between([scales_fit.min(), scales_fit.max()], p_mc-ACC_SCALE, p_mc + ACC_SCALE, color='blue', alpha=0.2, zorder=49, label=r'Allowed $p_{\mathrm{reco.}}$: $p_{\mathrm{true}}\pm$'+f'{ACC_SCALE} MeV/c')
    plt.fill_between([scale_center-delta_scale, scale_center+delta_scale], means_fit.min(), means_fit.max(), color='green', alpha=0.3, zorder=50, label=f'Allowed {sol} Scale: [{scale_center-delta_scale:0.5f}, {scale_center+delta_scale:0.5f}]')
    # plt.fill_between([scale_center-delta_scale, scale_center+delta_scale], p_mc-ACC_SCALE, p_mc + ACC_SCALE, color='green', alpha=0.4, label='Allowed DS Scale')
    l = plt.legend()
    l.set_zorder(55)
    plt.xticks(scales_ticks)
    plt.xlabel(xtitle)
    plt.ylabel('Mean Reconstructed Momentum [MeV/c]')
    plt.title(nice_name)
    fig.tight_layout()
    fig.savefig(plot_file_pre+f'mean_reco_mom_vs_{sol}_scale_{file_suffix}{file_suffix2}.pdf')
    fig.savefig(plot_file_pre+f'mean_reco_mom_vs_{sol}_scale_{file_suffix}{file_suffix2}.png')

    # std vs scale
    fig = plt.figure()
    plt.scatter(scales, stds, s=35, zorder=110, label='StDev. (point)')
    if file_suffix=='Mau13_TSOff':
        stds_fit, m_s, b_s, chi2red_s = fit_linear(scales, stds, np.ones_like(scales))
        plt.plot(scales, stds_fit, '-.', c='gray', zorder=100, label=f'\nFit: stddev = {m_s:0.3f} * scale + {b_s:0.3f}\n'+r'$\chi^2_{\mathrm{red.}}=$'+f'{chi2red_s:0.4f}\n')
        plt.legend()
    elif file_suffix=='Mau13_DSOff':
        if file_suffix2 == '':
            stds_fit, a_s, b_s, c_s, chi2red_s = fit_quadratic(scales[1:], stds[1:], np.ones_like(scales[1:]))
            plt.plot(scales[1:], stds_fit, '-.', c='gray', zorder=100, label=f'\nFit: stddev = {a_s:0.4f} * scale^2 + {b_s:0.4f} * scale + {c_s:0.4f}\n'+r'$\chi^2_{\mathrm{red.}}=$'+f'{chi2red_s:0.6f}\n')
        else:
            stds_fit, a_s, b_s, c_s, chi2red_s = fit_quadratic(scales, stds, np.ones_like(scales))
            inds = scales.argsort()
            plt.plot(scales[inds], stds_fit[inds], '-.', c='gray', zorder=100, label=f'\nFit: stddev = {a_s:0.4f} * scale^2 + {b_s:0.4f} * scale + {c_s:0.4f}\n'+r'$\chi^2_{\mathrm{red.}}=$'+f'{chi2red_s:0.6f}\n')
        plt.legend()
    plt.xticks(scales_ticks)
    plt.xlabel(xtitle)
    plt.ylabel('StdDev. Reconstructed Momentum [MeV/c]')
    plt.title(nice_name)
    fig.tight_layout()
    fig.savefig(plot_file_pre+f'stddev_reco_mom_vs_{sol}_scale_{file_suffix}{file_suffix2}.pdf')
    fig.savefig(plot_file_pre+f'stddev_reco_mom_vs_{sol}_scale_{file_suffix}{file_suffix2}.png')


plot_mean_scatter(scales_dis_str, scales, scales_ticks, file_suffix2='')
# plot_mean_scatter(scales_fine_str, scales_new, scales_new_ticks, file_suffix2='_fine')
plot_mean_scatter(scales_finer_str, scales_new, scales_new_ticks, file_suffix2='_finer')
