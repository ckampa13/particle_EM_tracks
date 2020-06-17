from numpy.polynomial.polynomial import Polynomial
from copy import deepcopy
import matplotlib.pyplot as plt
from .particle import trajectory_solver
from .tools import *
from .plotting import config_plots
config_plots()

def dist(sol_point, test_point):
    sol_point = sol_point[:3]
    return np.sqrt(np.sum((sol_point - test_point)**2))

def impact_param_point(sol_test, meas_point, alpha=1e-18, max_its=50, delta=0.001, plot_dists=False):
    # sol_test is solve_ivp solution object. Note must use dense=True to get a continuous solution
    # meas_point is a length 3 np.array denoting where our hypothesis track should pass through
    def g(t):
        return dist(sol_test.sol(t), meas_point)
    z_mask = np.array([False])
    i = 0
    while z_mask.sum() == 0:
        if i != 0:
            delta *= 2
        z_mask = (sol_test.y[2] > meas_point[2] - delta) & (sol_test.y[2] < meas_point[2] + delta)
        i += 1
    L = z_mask.sum()
    t0 = sol_test.t[z_mask][L // 2 - 1]
    wh, ch = gradient_descent_dimalpha(g, alpha, max_its, t0)
    tf = wh[ch.argmin()]
    if plot_dists:
        dists = np.array([g(t_) for t_ in sol_test.t])
        fig = plt.figure()
        plt.scatter(sol_test.t, dists, s=1, label='Data')
        plt.plot([t0, t0], [0., dists.max()], 'g-.', linewidth=1, label='Initial guess')
        plt.plot([tf, tf], [0., dists.max()], 'r--', linewidth=2, label='Optimized value')
        plt.xlabel('time [s]')
        plt.ylabel('distance between [m]')
        plt.legend()
    closest_point = sol_test.sol(tf)[:3]
    impact_param = dist(closest_point, meas_point)
    if plot_dists:
        return impact_param, closest_point, fig
    return impact_param, closest_point

def impact_params_in_tracker(sol_test, meas_solver, N_meas=40, tracker=True):
    if tracker:
        df = meas_solver.dataframe.query("z >= 8.41 & z <= 11.66").copy()
        df.reset_index(drop=True, inplace=True)
    else:
        df = meas_solver.dataframe

    stride = len(df) // N_meas
    df = df[::stride]
    impact_params = []
    closest_points = []
    for row in df.itertuples():
        ip, cp = impact_param_point(sol_test, np.array([row.x, row.y, row.z]))
        impact_params.append(ip)
        closest_points.append(cp)
    return np.array(impact_params), np.array(closest_points), df

def global_chi2_ptests(test_ps, test_B_func, meas_solver, sigma=1.):
    sum_of_impacts2 = []
    for p in test_ps:
        ic_ = deepcopy(meas_solver.init_conds)
        ic_.p0 = p
        test_solver = trajectory_solver(ic_, B_func=test_B_func, bounds=bounds_Mu2e)
        sol_test = test_solver.solve_trajectory(atol=1e-8, rtol=1e-8, dense=True)
        ips, cps, df_meas = impact_params_in_tracker(sol_test, meas_solver)
        sum_of_impacts2.append(np.sum(ips**2)/sigma**2)
    return np.array(sum_of_impacts2)

def momentum_analysis_global_chi2(meas_solver, analysis_B_func, B_name='Mau 13 (nominal)', sigma=1., delta=.2, N_hyp=7, plot_chi2=False):
    # use circle fit analysis mean to determine test_ps
    meas_solver.analyze_trajectory(step=25, stride=1, query="z >= 8.41 & z <= 11.66", B=analysis_B_func)
    circle_fit_mom = meas_solver.df_reco.p.mean()
    pmin = circle_fit_mom - delta
    pmax = circle_fit_mom + delta
    ps_test = np.linspace(pmin, pmax, N_hyp)
    # run particles for each hypothesis momentum
    sum_of_impacts2 = global_chi2_ptests(test_ps=ps_test, test_B_func=analysis_B_func,
                                         meas_solver=meas_solver, sigma=sigma)
    # fit quadratic to calculated chi2 values
    c, b, a = Polynomial.fit(ps_test, sum_of_impacts2, 2).convert().coef
    # use quadratic parameters to get momentum estimate
    chi2_fit_mom = - b / (2 * a)
    # plot (if selected)
    if plot_chi2:
        ptrue = meas_solver.init_conds.p0
        ps_quad = np.linspace(pmin, pmax, 50)
        y_quad = a * ps_quad**2 + b * ps_quad + c
        fig = plt.figure()
        plt.plot(ps_quad, y_quad, c='gray', label='Quadratic Fit')
        plt.scatter(ps_test, sum_of_impacts2, c='black', label='Calculated '+r"$\chi^2$", zorder=100)
        ymax = np.max(sum_of_impacts2)
        plt.plot([ptrue, ptrue], [0., ymax], 'r-', linewidth=2,
                 label=f'True Momentum ={ptrue:.3f} MeV/c', zorder=90)
        plt.plot([chi2_fit_mom, chi2_fit_mom], [0., ymax], 'g--',
                 label='Momentum '+r'$(\chi^2) =$'+f'{chi2_fit_mom:.3f} MeV/c', zorder=92)
        plt.plot([circle_fit_mom, circle_fit_mom], [0., ymax], 'b--',
                 label=f'Momentum (circle) = {circle_fit_mom:.3f} MeV/c', zorder=91)
        plt.xlabel(r"$p$"+" [MeV/c]")
        plt.ylabel(r"$\chi^2$")
        plt.title('Global '+r'$\chi^2$'+f' Momentum Fit: {B_name}')
        l = plt.legend(loc='upper right')
        l.set_zorder(110)
        return chi2_fit_mom, circle_fit_mom, fig
    # return results
    return chi2_fit_mom, circle_fit_mom
