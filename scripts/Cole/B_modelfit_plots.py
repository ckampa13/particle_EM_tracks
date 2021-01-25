import os
import pandas as pd
import plotly.io as pio

from hallprobecalib.hpcplots import histo

# datadir = '/home/ckampa/data/pickles/distortions/linear_gradient/'
# plotdir = '/home/ckampa/data/plots/distortions/linear_gradient/run_04/LHelix_reco/Model_fit/'
datadir = '/home/ckampa/data/pickles/emtracks/mid_hp_bias_up/'
# plotdir = '/home/ckampa/data/plots/emtracks/mid_hp_bias_up/Model_fit/'
plotdir = '/home/ckampa/data/plots/emtracks/hp_bias/single_up/'

# dist_name = 'Mau13_ModelFit'
# dist_name = 'mid_hp_bias_up'
dist_name = 'all_hp_bias_up'
file_suffix = dist_name

nums = [0, 1, 2, 3, 4]

# df_run = pd.read_pickle(datadir+f'run_04/MC_sample_plus_reco_{file_suffix}.pkl')
df_run = pd.read_pickle(datadir+f'MC_sample_plus_reco_{file_suffix}.pkl')

# df_run.loc[:, 'Momentum Residual (Mau13 reco.)'] = df_run['mom_nom'] - df_run['p_mc']
# df_run.loc[:, 'Momentum Residual (Pollack fit reco.)'] = df_run['mom_dis_fit'] - df_run['p_mc']
df_run.loc[:, 'Momentum Residual (Pollack fit reco.)'] = df_run['mom_nom'] - df_run['p_mc']
for num in nums:
    df_run.loc[:, f'Momentum Residual<br>(Hall {num}: 1e-3 Rel. Bias fit reco.)'] = df_run[f'mom_dis_hp_{num}'] - df_run['p_mc']

# N = 75
N = 35
# N2 = 200
N2 = 75

plot_file_pre = plotdir

# fig = histo([df_run['Momentum Residual (Mau13 reco.)']], bins=N, inline=False, show_plot=False)
fig = histo([df_run['Momentum Residual (Pollack fit reco.)']], bins=N, inline=False, show_plot=False)
fig.update_layout(xaxis=dict(title=r"$\text{Residual }(p_{\text{reco}} - p_{\text{MC}}) [\text{MeV } c^{-1}]$"),title="Event-by-event Residuals")
pio.write_image(fig, plot_file_pre+'hist_residuals_nom.pdf')
pio.write_image(fig, plot_file_pre+'hist_residuals_nom.png')

# fig = histo([df_run['Momentum Residual (Pollack fit reco.)']], bins=N, inline=False, show_plot=False)
for num in nums:
    fig = histo([df_run[f'Momentum Residual<br>(Hall {num}: 1e-3 Rel. Bias fit reco.)']], bins=N, inline=False, show_plot=False)
    fig.update_layout(xaxis=dict(title=r"$\text{Residual }(p_{\text{reco}} - p_{\text{MC}}) [\text{MeV } c^{-1}]$"),title=f"Event-by-event Residuals: Hall Probe {num} [0 smallest radius, 4 largest radius]")
    pio.write_image(fig, plot_file_pre+f'hist_residuals_hp_{num}.pdf')
    pio.write_image(fig, plot_file_pre+f'hist_residuals_hp_{num}.png')

# fig = histo([df_run['Momentum Residual (Mau13 reco.)'], df_run['Momentum Residual (Pollack fit reco.)']], bins=N2, same_bins=True, inline=False, show_plot=False)
fig = histo([df_run['Momentum Residual (Pollack fit reco.)']]+[df_run[f'Momentum Residual<br>(Hall {num}: 1e-3 Rel. Bias fit reco.)'] for num in nums], bins=N2, same_bins=True, inline=False, show_plot=False)
fig.update_layout(xaxis=dict(title=r"$\text{Residual }(p_{\text{reco}} - p_{\text{MC}}) [\text{MeV } c^{-1}]$"),title="Event-by-event Residuals")
pio.write_image(fig, plot_file_pre+'residual_mean_comparison.pdf')
pio.write_image(fig, plot_file_pre+'residual_mean_comparison.png')
