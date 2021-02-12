import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from emtracks.plotting import config_plots

config_plots()

ddir = "/home/ckampa/data/"
pkldir = ddir + 'pickles/emtracks/pion_degrader_tandip/'
# plotdir = ddir + 'plots/emtracks/pion_degrader_tandip/70perc/run03/'
# plotdir = ddir + 'plots/emtracks/pion_degrader_tandip/70perc/run04/'
# plotdir = ddir + 'plots/emtracks/pion_degrader_tandip/70perc/run05/'
# plotdir = ddir + 'plots/emtracks/pion_degrader_tandip/70perc/run06/'
# plotdir = ddir + 'plots/emtracks/pion_degrader_tandip/70perc/run07/'
# plotdir = ddir + 'plots/emtracks/pion_degrader_tandip/70perc/run08/'
# plotdir = ddir + 'plots/emtracks/pion_degrader_tandip/70perc/run09/'
# plotdir = ddir + 'plots/emtracks/pion_degrader_tandip/70perc/run10/'
plotdir = ddir + 'plots/emtracks/pion_degrader_tandip/70perc/run11/'
# filenames
# pklfile = pkldir+'degrader_tandip_df_run02.pkl'
# pklfile = pkldir+'degrader_tandip_df_run03.pkl'
# pklfile = pkldir+'degrader_tandip_df_run04.pkl'
# pklfile = pkldir+'degrader_tandip_df_run05.pkl'
# pklfile = pkldir+'degrader_tandip_df_run06.pkl'
# pklfile = pkldir+'degrader_tandip_df_run07.pkl'
# pklfile = pkldir+'degrader_tandip_df_run08.pkl'
# pklfile = pkldir+'degrader_tandip_df_run09.pkl'
# pklfile = pkldir+'degrader_tandip_df_run10.pkl'
pklfile = pkldir+'degrader_tandip_df_run11.pkl'
# pklfile_new = pkldir+'degrader_tandip_df_run03_plots.pkl'
# pklfile_new = pkldir+'degrader_tandip_df_run04_plots.pkl'
# pklfile_new = pkldir+'degrader_tandip_df_run05_plots.pkl'
# pklfile_new = pkldir+'degrader_tandip_df_run06_plots.pkl'
# pklfile_new = pkldir+'degrader_tandip_df_run07_plots.pkl'
# pklfile_new = pkldir+'degrader_tandip_df_run08_plots.pkl'
# pklfile_new = pkldir+'degrader_tandip_df_run09_plots.pkl'
# pklfile_new = pkldir+'degrader_tandip_df_run10_plots.pkl'
pklfile_new = pkldir+'degrader_tandip_df_run11_plots.pkl'

# tracker cut
Rmin = 0. # 0.3769 # m
# bins
bins = np.linspace(0, 2, 101)
# bins = np.linspace(0, 2, 201)

# load results
df = pd.read_pickle(pklfile)
df.eval('px = p * sin(theta) * cos(phi)', inplace=True)
df.eval('py = p * sin(theta) * sin(phi)', inplace=True)
df.eval('pz = p * cos(theta)', inplace=True)
df.eval('pT = (px**2 + py**2)**(1/2)', inplace=True)
df.eval('r = (x**2 + y**2)**(1/2)', inplace=True)
df.eval('phi = arctan2(y,x)', inplace=True)
df.eval('p0phi = -px*sin(phi)+py*cos(phi)', inplace=True)
phi_max = np.max(np.abs(df['p0phi']))
R_max = np.max(np.abs(df['r']))
# df.eval(f'p0phi_')
# df.eval(f'scaled_p0phi_r = p0phi/{phi_max} * r/{R_max}', inplace=True)
# df.eval(f'scaled_p0phi_r = p0phi/pT * r/{R_max}', inplace=True)
df.eval(f'scaled_p0phi_r = p0phi/pT + r', inplace=True) # fix
df.eval('p0r = px*cos(phi)+py*sin(phi)', inplace=True)
# bins_r = np.linspace(0, 0.175, 176)
# n, bins = np.histogram(df['r'].values, bins=bins_r)
# df.loc[df.tand_Mau9_70 < 0, 'tand_Mau9_70'] = -1
df = df[df['tand_Mau9_70'] >= 0].copy()
N_gen = len(df)


# save pickle with new columns
df.to_pickle(pklfile_new)

# hard coded peak locations
q1 = '(tand_Mau9_70 < 1.27)'
q2 = '(tand_Mau9_70 >= 1.27) & (tand_Mau9_70 < 1.56)'
f = '(tand0_Mau9_70 > 0)'
b = '(tand0_Mau9_70 < 0)'


# a few functions to keep code DRY
# def get_data(df, var, query=None, Rmin=Rmin):
def get_data(df, var, query=None):
    if query is None:
        df_ = df
    else:
        df_ = df.query(query)
    # data = df_.query(f'Rmax_{name} >= {Rmin}')[f'{var}'].values
    data = df_[var].values
    return data

# reference label
# label_temp = '{0}\n' + r'$\mu = {1:.3E}$'+ '\n' + 'std' + r'$= {2:.3E}$' + '\n' +  'Integral: {3}\n' + 'Underflow: {4}\nOverflow: {5}'

def get_label(name, data, bins):
    over = (data > np.max(bins)).sum()
    under = (data < np.min(bins)).sum()
    # label = f'{name}\n' + rf'$\mu: {np.mean(tand):.3E}$' + '\n' + rf'$\sigma: {np.std(tand):.3E}$' + '\n' + f'Integral: {len(tand)}\nUnderflow: {under}\nOverflow: {over}'
    data = data[(data <= np.max(bins)) & (data >= np.min(bins))]
    mean = f'{np.mean(data):.3E}'
    std = f'{np.std(data):.3E}'
    # n = 15
    # label = f'mean: {mean:>15}' + '\n' + f'stddev: {std:>15}' + '\n' + f'Integral: {len(data)-over-under:>17}\nUnderflow: {under:>16}\nOverflow: {over:>16}'
    label = f'mean: {mean:>15}' + '\n' + f'stddev: {std:>15}' + '\n' + f'Integral: {len(data):>17}\nUnderflow: {under:>16}\nOverflow: {over:>16}'
    return label

def make_plot_hist(df, name='Mau9 70%', var='tand_Mau9_70', xl=r'$\tan(\mathrm{dip})$', query=None, queryn='full', bins=bins, legendloc='upper right', reweight=False):
    data = get_data(df, var, query)
    fig = plt.figure()
    if reweight:
        n_full, _ = np.histogram(get_data(df, var, '(tand0_Mau9_70 > 0)'), bins=bins)
        n_q, _ = np.histogram(data, bins=bins)
        ws = n_q / n_full
        ws = n_q / bins[:-1]
        ws[np.isnan(ws)] = 0.
        ws[np.isinf(ws)] = 0.
        plt.hist(bins[:-1], bins=bins, weights= ws, histtype='step', linewidth=1.5, label=get_label(name, data, bins))
        plt.ylabel('Events (reweighted)')
    else:
        plt.hist(data, bins=bins, histtype='step', linewidth=1.5, label=get_label(name, data, bins))
        plt.ylabel('Events')
    # if var == 'r':
    #     a = 0.1727
    #     b = 2/a
    #     m = b/a
    #     xs = np.linspace(0,a, 100)
    #     ys = m*xs
    #     plt.plot(xs,ys,'r--', label='Expected PDF')
    plt.xlabel(xl)
    # plt.ylabel('Probability Density')
    if query is None:
        plt.title(f'{name}: 69.8 MeV/c Isotropic e+ Uniformly Generated in Pion Degrader\n'+r'$N_{\mathrm{gen}} = 1\times10^4$')#+'; Tracker Cut ('+r'$R_{\mathrm{max}} >= $'+rf'${Rmin}$'+' [m])')
    else:
        plt.title(f'{name}: 69.8 MeV/c Isotropic e+ Uniformly Generated in Pion Degrader\n'+r'$N_{\mathrm{gen}} = 1\times10^4$'+f'; {query}') # +'; Tracker Cut ('+r'$R_{\mathrm{max}} >= $'+rf'${Rmin}$'+' [m])')
    plt.legend(loc=legendloc)
    fig.savefig(plotdir+f'{var}_{queryn}_degrader_hist.pdf')
    fig.savefig(plotdir+f'{var}_{queryn}_degrader_hist.png')

def make_plot_scatter(df, name='Mau9 70%', x='tand0_Mau9_70', y='tand_Mau9_70', xl=r'$\tan(\mathrm{dip})$ (positron birth)', yl=r'$\tan(\mathrm{dip})$ (Tracker start)', query=f, queryn='forward', legendloc='upper left'):
    data_x = get_data(df, x, query)
    data_y= get_data(df, y, query)
    fig = plt.figure()
    plt.scatter(data_x, data_y, s=1)
    plt.xlabel(xl)
    plt.ylabel(yl)
    if query is None:
        plt.title(f'{name}: 69.8 MeV/c Isotropic e+ Uniformly Generated in Pion Degrader\n'+r'$N_{\mathrm{gen}} = 1\times10^4$')
    else:
        plt.title(f'{name}: 69.8 MeV/c Isotropic e+ Uniformly Generated in Pion Degrader\n'+r'$N_{\mathrm{gen}} = 1\times10^4$'+f'; {query}')

    # plt.show()

    fig.savefig(plotdir+f'{y}_vs_{x}_{queryn}_degrader_scatter.pdf')
    fig.savefig(plotdir+f'{y}_vs_{x}_{queryn}_degrader_scatter.png')


if __name__=='__main__':
    # names = [f'Mau{i}' for i in [9]]
    # names = [f'Mau{i}' for i in [9, 10, 12, 13]]
    # names = ['Mau9 70%']
    vars_ = ['tand_Mau9_70', 'tand_Mau9_70', 'tand_Mau9_70', 'r', 'r', 'r', 'r', 'r', 'r',
             'p0phi', 'p0phi','p0phi', 'scaled_p0phi_r', 'scaled_p0phi_r', 'scaled_p0phi_r']
    # queries = [None, None, '(tand_Mau9_70 > 0.91) & (tand_Mau9_70 < 1.08)', '(tand_Mau9_70 > 1.08) & (tand_Mau9_70 < 1.21)']
    queries = [None, b, f, None, q1, q2, f, f+'&'+q1, f+'&'+q2,
               f, f+'&'+q1, f+'&'+q2,
               f, f+'&'+q1, f+'&'+q2]
    queriens = ['full', 'backwards', 'forward', 'full', 'peak1', 'peak2', 'forward_full','forward_peak1','forward_peak2',
                'forward_full','forward_peak1','forward_peak2',
                'forward_full','forward_peak1','forward_peak2',]
    xlabs = [r'$\tan(\mathrm{dip})$', r'$\tan(\mathrm{dip})$', r'$\tan(\mathrm{dip})$', r'$r$ (positron birth) [m]', r'$r$ (positron birth) [m]', r'$r$ (positron birth) [m]', r'$r$ (positron birth) [m]', r'$r$ (positron birth) [m]', r'$r$ (positron birth) [m]',
             r'$p_{0 \phi}$ [MeV] (positron birth)', r'$p_{0 \phi}$ [MeV] (positron birth)', r'$p_{0 \phi}$ [MeV] (positron birth)',
             r'$\frac{p_{0 \phi}}{p_{T}} \times \frac{r}{r_{max.}}$',r'$\frac{p_{0 \phi}}{p_{T}} \times \frac{r}{r_{max.}}$',r'$\frac{p_{0 \phi}}{p_{T}} \times \frac{r}{r_{max.}}$',]
             # r'$\frac{p_{0 \phi}}{p_{0 \phi max.}} \times \frac{r}{r_{max.}}$',r'$\frac{p_{0 \phi}}{p_{0 \phi max.}} \times \frac{r}{r_{max.}}$',r'$\frac{p_{0 \phi}}{p_{0 \phi max.}} \times \frac{r}{r_{max.}}$',]
    bins_tan = np.linspace(0, 2, 101)
    # bins_r = np.linspace(0, 0.175, 176)
    # bins_r = np.linspace(0, 0.173, 174)
    bins_r = np.linspace(0, 0.173, int(174/2))
    # bins_r = np.linspace(0, 0.175, int(176/2))
    bins_pp = np.linspace(-70, 70, 70)
    # bins_pp = np.linspace(-69, 69, 139)
    # bins_scaled = np.linspace(-1, 1, 101)
    bins_scaled = np.linspace(-2, 2, 101)
    binss = [bins_tan, bins_tan, bins_tan, bins_r, bins_r, bins_r, bins_r, bins_r, bins_r,
             bins_pp, bins_pp, bins_pp, bins_scaled, bins_scaled, bins_scaled]
    llocs = ['upper right',  'upper right', 'upper right', 'upper left', 'upper left', 'upper left', 'upper left', 'upper right', 'upper right',
             'best', 'best', 'best', 'best','best','best']
    reweights = [False, False, False, False, True, True, False, True, True,
                 False, False, False, False, False, False]

    for v, q, qn, xl, b, ll, rw in zip(vars_, queries, queriens, xlabs, binss, llocs, reweights):
        make_plot_hist(df, var=v, xl=xl, query=q, queryn=qn, bins=b, legendloc=ll, reweight=rw)

    # scatter plots
    make_plot_scatter(df, query='(tand0_Mau9_70 > -1.3) & (tand_Mau9_70 < 2.)', queryn='all')
    make_plot_scatter(df)
    make_plot_scatter(df, query=f+'& (tand_Mau9_70 < 2.0)', queryn='full')
    # make_plot_scatter(df, query='(tand0_Mau9_70 > 0) & (tand_Mau9_70 < 2.0)', queryn='forward-tracker')
    make_plot_scatter(df, query=f+'&'+q1, queryn='peak1')
    make_plot_scatter(df, query=f+'&'+q2, queryn='peak2')
    make_plot_scatter(df, query=f+'&'+q2, queryn='peak2', x='p0phi', xl=r'$p_{0 \phi}$ (positron birth)')
    make_plot_scatter(df, query='(tand0_Mau9_70 > -1.3) & (tand_Mau9_70 < 2.)', queryn='full', x='theta', xl=r'$p_{\theta}$ (positron birth)')
    make_plot_scatter(df, query='(tand0_Mau9_70 > -10) & (tand0_Mau9_70 < 10) & (tand_Mau9_70 < 10.)', queryn='full2',)# x='theta', xl=r'$p_{\theta}$ (positron birth)')

    # fig,ax

