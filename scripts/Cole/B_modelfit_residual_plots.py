import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from emtracks.particle import trajectory_solver
from emtracks.plotting import config_plots
config_plots()
plt.rcParams['figure.figsize'] = [13, 11] # larger figures
plt.rcParams.update({'font.size': 18.0})   # increase plot font size

num = 4

# plot_dir = '/home/ckampa/data/plots/distortions/linear_gradient/run_04/LHelix_reco/Model_fit/'
# plot_dir = '/home/ckampa/data/plots/emtracks/mid_hp_bias_up/Model_fit/'
plot_dir = '/home/ckampa/data/plots/emtracks/hp_bias/single_up/BField_Fit_Residuals/'

# load dfs
# full fit
# df_full = pd.read_pickle('/home/ckampa/data/Bmaps/Mau13_standard_fit_df.p')
# df_full = pd.read_pickle('/home/ckampa/data/Bmaps/Mau13_standard_fit_df_cutz.p')
# df_full = pd.read_pickle('/home/ckampa/data/Bmaps/Mau13_middle_hp_bias_up_fit_df.p')
df_full = pd.read_pickle(f'/home/ckampa/data/Bmaps/hp_bias/Mau13_hp_{num}_bias_up_fit_df.p')
df_full.eval('B = (Bx**2+By**2+Bz**2)**(1/2)', inplace=True)
df_full.eval('B_fit = (Bx_fit**2+By_fit**2+Bz_fit**2)**(1/2)', inplace=True)
df_full.eval('B_res = B - B_fit', inplace=True)
df_full.eval('Bx_res = Bx - Bx_fit', inplace=True)
df_full.eval('By_res = By - By_fit', inplace=True)
df_full.eval('Bz_res = Bz - Bz_fit', inplace=True)
df_full = df_full.query('Z >= 4.25  & Z <= 14. & R <= 0.8').copy()
# df_full = df_full.query('R <= 0.8').copy()
# tracker fit
# old: load new pickle
# df_fit = pd.read_pickle('/home/ckampa/data/Bmaps/Mau13_standard_tracker_fit_df.p')
# df_fit.eval('B = (Bx**2+By**2+Bz**2)**(1/2)', inplace=True)
# df_fit.eval('B_fit = (Bx_fit**2+By_fit**2+Bz_fit**2)**(1/2)', inplace=True)
# df_fit.eval('B_res = B - B_fit', inplace=True)
# df_fit.eval('Bx_res = Bx - Bx_fit', inplace=True)
# df_fit.eval('By_res = By - By_fit', inplace=True)
# df_fit.eval('Bz_res = Bz - Bz_fit', inplace=True)
# df_fit = df_fit.query('R <= 0.8').copy()
# new: query full df
df_fit = df_full.query('R <= 0.8 & Z >= 8.41 & Z <= 11.66')
# tracks, tracker
# df_run = pd.read_pickle('/home/ckampa/data/pickles/distortions/linear_gradient/run_04/B_Residuals_Mau13_Fit.pkl')
# df_run = pd.read_pickle('/home/ckampa/data/pickles/emtracks/mid_hp_bias_up/B_Residuals_Mau13_Fit.pkl')
df_run = pd.read_pickle(f'/home/ckampa/data/pickles/emtracks/mid_hp_bias_up/B_Residuals_Mau13_Fit_hp_{num}.pkl')
# tracks, tracker, straws
# df_run_straws = pd.read_pickle('/home/ckampa/data/pickles/distortions/linear_gradient/run_04/B_Residuals_Mau13_Fit_Straws.pkl')
# df_run_straws = pd.read_pickle('/home/ckampa/data/pickles/emtracks/mid_hp_bias_up/B_Residuals_Mau13_Fit_Straws.pkl')
df_run_straws = pd.read_pickle(f'/home/ckampa/data/pickles/emtracks/mid_hp_bias_up/B_Residuals_Mau13_Fit_hp_{num}_Straws.pkl')

def make_plot(df, file_suffix='_tracker', title_suffix='Tracker Region'):
    label_temp = r'$\mu = {0:.3E}$'+ '\n' + 'std' + r'$= {1:.3E}$' + '\n' +  'Integral: {2}'
    print("Generating plots:"+file_suffix)
    # simple histograms
    N_bins = 200
    lsize = 16
    xmin = df[['Bx_res','By_res','Bz_res','B_res']].min().min()
    xmax = df[['Bx_res','By_res','Bz_res','B_res']].max().max()+1e-5
    bins = np.linspace(xmin, xmax, N_bins+1)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(df['Bx_res'], bins=bins, label=label_temp.format(df['Bx_res'].mean(), df['Bx_res'].std(), len(df)))
    axs[0, 0].set(xlabel=r"$\Delta B_x$"+" [Gauss]", ylabel="Count")
    axs[0, 0].legend(prop={'size': lsize})
    axs[0, 1].hist(df['By_res'], bins=bins, label=label_temp.format(df['By_res'].mean(), df['By_res'].std(), len(df)))
    axs[0, 1].set(xlabel=r"$\Delta B_y$"+" [Gauss]", ylabel="Count")
    axs[0, 1].legend(prop={'size': lsize})
    axs[1, 0].hist(df['Bz_res'], bins=bins, label=label_temp.format(df['Bz_res'].mean(), df['Bz_res'].std(), len(df)))
    axs[1, 0].set(xlabel=r"$\Delta B_z$"+" [Gauss]", ylabel="Count")
    axs[1, 0].legend(prop={'size': lsize})
    axs[1, 1].hist(df['B_res'], bins=bins, label=label_temp.format(df['B_res'].mean(), df['B_res'].std(), len(df)))
    axs[1, 1].set(xlabel=r"$\Delta |B|$"+" [Gauss]", ylabel="Count")
    axs[1, 1].legend(prop={'size': lsize})
    # title_main=f'Mau 13 Subtraction with Mau 10 PS+TS: Field Difference from Mau 13\n{DS_frac:0.3f}xDS, {TS_frac:0.3f}x(PS+TS)\n'+r"$(\Delta B = B_{\mathregular{Mau10\ comb.}} - B_{\mathregular{Mau13}})$"
    title_main=f'Hall Probe {num} Biased: Model Fit Residuals ('+ r'$\Delta B = B_\mathrm{data} - B_\mathrm{fit}$'+'):\n'
    fig.suptitle(f'{title_main}{title_suffix}')
    fig.tight_layout(rect=[0,0,1,0.9])
    # plot_file = plot_dir+f'Mau13_{DS_frac:0.3f}xDS_{TS_frac:0.3f}xPS-TS_Comparison_Hists'+file_suffix
    plot_file = plot_dir+f'Mau13_fit_residuals_hp_{num}'+file_suffix
    fig.savefig(plot_file+'.pdf')
    fig.savefig(plot_file+'.png')
    print("Generating plots complete.\n")

    return fig, axs

# run plotting function
make_plot(df_full, '_full', 'Grid in Entire DS (4.25 <= Z <= 14 m) (R <= 0.8 m)')
make_plot(df_fit, '_tracker', 'Grid in Tracker Region')
make_plot(df_run, '_tracker_tracks', 'Signal e- Tracks in Tracker Region')
make_plot(df_run_straws, '_tracker_tracks_straws', 'Signal e- Tracks in Tracker Region (40 cm <= R <= 70 cm)')
