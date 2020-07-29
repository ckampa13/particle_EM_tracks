import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# significance function
def signal_significance(s, b):
    # S = sqrt(2 * ((s+b) * ln(1+s/b) -s))
    if b == 0:
        # return -1
        b = 1e-10
    S = (2*((s+b)*np.log(1+s/b)-s))**(1/2)
    return S

Rmue = 2e-16
p_low_cut = 103.85
p_hi_cut = 104.90
window_width = p_hi_cut - p_low_cut
bin_width=0.05
N_OTHER = 0.01415 * window_width/bin_width # from plot digitization
MeVc_per_Scale = 104.6452 # from mean momentum vs DS scale

plot_file_pre = plotdir+f'run_04/LHelix_reco/{dist_name}/{sol}_scale/'

# load digitization file
digit_file = '/home/ckampa/data/root/cd3_ce_and_background_digitized.csv'
df_dig = pd.read_csv(digit_file, names=["mom_DIO", "N_DIO", "mom_CE", "N_CE"], skiprows=2)
mom_CE, N_CE_dig = df_dig[["mom_CE", "N_CE"]].values.T
mom_DIO, N_DIO_dig = df_dig[["mom_DIO","N_DIO"]].dropna().values.T


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

def sig_vs_window(p_lo=p_low_cut, p_hi=p_hi_cut):
    mask_DIO = (mom_DIO>=p_lo) & (mom_DIO<=p_hi)
    mask_CE = (mom_CE>=p_lo) & (mom_CE<=p_hi)
    N0_DIO = N_DIO_dig[mask_DIO][0]*(mom_DIO[mask_DIO][0] - p_lo)/bin_width
    Nf_DIO = N_DIO_dig[mask_DIO][-1]*(p_hi - mom_DIO[mask_DIO][-1])/bin_width
    N0_CE = N_CE_dig[mask_CE][0]*(mom_CE[mask_CE][0] - p_lo)/bin_width
    Nf_CE = N_CE_dig[mask_CE][-1]*(p_hi - mom_CE[mask_CE][-1])/bin_width
    N_DIO_window = N_DIO_dig[mask_DIO][1:-1].sum() + N0_DIO + Nf_DIO
    N_CE_window = N_CE_dig[mask_CE][1:-1].sum() + N0_CE + Nf_CE
    sig = signal_significance(N_CE_window, N_DIO_window)
    return sig, N_CE_window, N_DIO_window

# Mu2eCCFCNoNtuples grid
p_lows = np.linspace(103.5, 104.,51)
p_his = np.linspace(104.75, 105.5, 76)
# coarse grid
# p_lows = np.linspace(103.3, 105.5, 23)
# p_his = np.linspace(103.3, 105.5, 23)
# fine grid
# p_lows = np.linspace(103.7, 103.9, 201)
# p_his = np.linspace(104.85, 105.1, 251)
PL, PH = np.meshgrid(p_lows, p_his)
PL_ = PL.flatten()
PH_ = PH.flatten()
sigs = []
N_CEs = []
N_DIOs = []
for pl, ph in zip(PL_,PH_):
    if pl>=ph:
        sigs.append(0)
        N_CEs.append(0)
        N_DIOs.append(0)
    else:
        s, c, d = sig_vs_window(pl,ph)
        sigs.append(s)
        N_CEs.append(c)
        N_DIOs.append(d)
        # sigs.append(sig_vs_window(pl,ph))
sigs = np.array(sigs)
N_CEs = np.array(N_CEs)
N_DIOs = np.array(N_DIOs)

sig_opt = sigs.max()
pl_opt = PL_[sigs.argmax()]
ph_opt = PH_[sigs.argmax()]

# SIGS = sigs.reshape(len(p_lows), len(p_his))
SIGS = sigs.reshape(len(p_his), len(p_lows))
CES = N_CEs.reshape(len(p_his), len(p_lows))
SESS = Rmue / CES
DIOS = N_DIOs.reshape(len(p_his), len(p_lows))

# plot!
fig = plt.figure()
ax = fig.add_subplot(111)
c = ax.pcolormesh(PL, PH, SIGS, cmap='gist_rainbow_r')
cb = fig.colorbar(c)
cb.ax.set_ylabel('Signal Significance')
# grid results
ax.plot([pl_opt, pl_opt],[p_his.min(), p_his.max()], '--', c='gray', label=f'Grid Max Significance:\n'+r'$p_{min}=$'+f'{pl_opt:0.3f} MeV/c, '+r'$p_{max}=$'+f'{ph_opt:0.3f} MeV/c')
ax.plot([p_lows.min(), p_lows.max()],[ph_opt, ph_opt], '--', c='gray')
# nominal results
ax.plot([p_low_cut, p_low_cut],[p_his.min(), p_his.max()], 'k-.', label=f'Nominal Window:\n'+r'$p_{min}=$'+f'{p_low_cut:0.3f} MeV/c, '+r'$p_{max}=$'+f'{p_hi_cut:0.3f} MeV/c')
ax.plot([p_lows.min(), p_lows.max()],[p_hi_cut, p_hi_cut], 'k-.')
ax.set_aspect('equal')

ax.set_xlabel(r'$p_{min}$')
ax.set_ylabel(r'$p_{max}$')
ax.set_xticks([103.5, 103.745, 103.99])
ax.set_yticks([104.75, 105, 105.25])

plt.legend()
fig.savefig(plot_file_pre+'significance_grid.pdf')
fig.savefig(plot_file_pre+'significance_grid.png')

# N_background
fig = plt.figure()
ax = fig.add_subplot(111)
# c = plt.contourf(PL, PH, DIOS, np.linspace(0.3, 1.2, 7))
# c = plt.pcolormesh(PL, PH, DIOS, cmap='gist_rainbow_r')
c = ax.pcolormesh(PL, PH, np.clip(DIOS,a_min=0.3,a_max=None), cmap='gist_rainbow_r', zorder=45)
cb = fig.colorbar(c)
cb.ax.set_ylabel('N background')
ax.set_aspect('equal')
ax.set_xlabel(r'$p_{min}$')
ax.set_ylabel(r'$p_{max}$')
ax.set_xticks([103.5, 103.745, 103.99])
ax.set_yticks([104.75, 105, 105.25])
# ax.set_axisbelow(False)
# ax.grid(zorder=50)
fig.savefig(plot_file_pre+'N_BG_grid.pdf')
fig.savefig(plot_file_pre+'N_BG_grid.png')

# N_SES
fig = plt.figure()
ax = fig.add_subplot(111)
# c = plt.contourf(PL, PH, DIOS, np.linspace(0.3, 1.2, 7))
c = ax.pcolormesh(PL, PH, SESS, cmap='gist_rainbow_r')
# c = ax.pcolormesh(PL, PH, np.clip(DIOS,a_min=0.3,a_max=None), cmap='gist_rainbow_r', zorder=45)
cb = fig.colorbar(c)
cb.ax.set_ylabel('SES')
ax.set_aspect('equal')
ax.set_xlabel(r'$p_{min}$')
ax.set_ylabel(r'$p_{max}$')
ax.set_xticks([103.5, 103.745, 103.99])
ax.set_yticks([104.75, 105, 105.25])
# ax.set_axisbelow(False)
# ax.grid(zorder=50)
fig.savefig(plot_file_pre+'SES_grid.pdf')
fig.savefig(plot_file_pre+'SES_grid.png')

# N_signal
fig = plt.figure()
# c = plt.contourf(PL, PH, CES, 7)
c = plt.contourf(PL, PH, CES, 7)
cb = fig.colorbar(c)
cb.ax.set_ylabel('N signal')
ax.set_xlabel(r'$p_{min}$')
ax.set_ylabel(r'$p_{max}$')
fig.savefig(plot_file_pre+'N_CE_grid.pdf')
fig.savefig(plot_file_pre+'N_CE_grid.png')

