import os
import pandas as pd
import plotly.io as pio

from hallprobecalib.hpcplots import histo

datadir = '/home/ckampa/data/pickles/distortions/linear_gradient/'
plotdir = '/home/ckampa/data/plots/distortions/linear_gradient/'

# for d in [plotdir, plotdir+'run_01/', plotdir+'run_02-10x_grad/']:
for d in [plotdir, plotdir+'run_04/',]:# plotdir+'run_02-10x_grad/']:
    if not os.path.exists(d):
        os.makedirs(d)

df_run = pd.read_pickle(datadir+'run_04/MC_sample_plus_reco.pkl')
# df_run = pd.read_pickle(datadir+'run_01/MC_sample_plus_reco.pkl')
# df_run2 = pd.read_pickle(datadir+'run_02-10x_grad/MC_sample_plus_reco.pkl')

df_run.loc[:, 'Residual Nominal'] = df_run['p_nom'] - df_run['p_mc']
df_run.loc[:, 'Residual Distorted'] = df_run['p_dis'] - df_run['p_mc']
df_run.loc[:, r'Standard Deviation Reco p (Nominal)'] = df_run['std_p_nom']
df_run.loc[:, r'Standard Deviation Reco p (Distorted)'] = df_run['std_p_dis']
# df_run2.loc[:, 'Residual Nominal'] = df_run2['p_nom'] - df_run2['p_mc']
# df_run2.loc[:, 'Residual Distorted'] = df_run2['p_dis'] - df_run2['p_mc']

N = 25

fig = histo([df_run['Residual Nominal'], df_run['Residual Distorted']], bins=N, inline=False, show_plot=False)
fig.update_layout(xaxis=dict(title=r"$\text{Residual }(\text{mean}(p_{\text{reco}}) - p_{\text{MC}}) [\text{MeV } c^{-1}]$"),title="Linear Gradient: 100 Gauss to 0 Gauss")
pio.write_image(fig, plotdir+'run_04/residual_mean_comparison_100_Gauss.pdf')
# pio.write_image(fig, plotdir+'run_01/residual_comparison_100_Gauss.pdf')

fig = histo([df_run['Standard Deviation Reco p (Nominal)'], df_run['Standard Deviation Reco p (Distorted)']], bins=N, inline=False, show_plot=False)
fig.update_layout(xaxis=dict(title=r"$\text{Standard Deviation along trajectory } [\text{MeV } c^{-1}]$"),title="Linear Gradient: 100 Gauss to 0 Gauss")
pio.write_image(fig, plotdir+'run_04/stddev_comparison_100_Gauss.pdf')

# fig = histo([df_run2['Residual Nominal'], df_run2['Residual Distorted']], bins=N, inline=False, show_plot=False)
# fig.update_layout(xaxis=dict(title=r"$\text{Residual }(p_{\text{reco}} - p_{\text{MC}}) [\text{MeV } c^{-1}]$"),title="Linear Gradient: 1000 Gauss to 0 Gauss")
# pio.write_image(fig, plotdir+'run_02-10x_grad/residual_comparison_1000_Gauss.pdf')
