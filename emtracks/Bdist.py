import numpy as np

def get_B_df_distorted(df, v="0", **kwargs):
    if v == "0":
        return dist0(df, **kwargs)

def dist0(df, **kwargs):
    df_ = df.copy()

    if 'Bzf' in kwargs.keys():
        Bzf = kwargs['Bzf']
    else:
        Bzf = 0.
    if 'Bz0' in kwargs.keys():
        Bz0 = kwargs['Bz0']
    else:
        Bz0 = 100. # Gauss
    if 'z0' in kwargs.keys():
        z0 = kwargs['z0']
    else:
        z0 = 3.239 # m
    if 'zf' in kwargs.keys():
        zf = kwargs['zf']
    else:
        zf = 14.139 # m

    slope = (Bzf - Bz0) / (zf - z0)
    intercept = (Bzf - slope * zf)

    Bx, By, Bz = linear_gradient(df_[['X','Y','Z']].values.T, slope, intercept)
    # print(Bx.mean(),By.mean(),Bz.mean())
    Br = np.sqrt(Bx**2 + By**2)
    Bphi = -Bx*np.sin(df_.Phi.values)+By*np.cos(df_.Phi.values)
    df_.eval('Bx = Bx + @Bx', inplace=True)
    df_.eval('By = By + @By', inplace=True)
    df_.eval('Bz = Bz + @Bz', inplace=True)
    df_.eval('Br = Br + @Br', inplace=True)
    df_.eval('Bphi = Bphi + @Bphi', inplace=True)

    return df_

def linear_gradient(pos, slope, intercept):
    x, y, z = pos
    r = (x**2 + y**2)**(1/2)
    phi = np.arctan2(y,x)
    Bz = slope * z + intercept
    Br = - (r / 2) * slope #  Br = - (r / 2) * (d Bz / d Br)
    Bx = Br * np.cos(phi)
    By = Br * np.sin(phi)
    return np.array([Bx, By, Bz])
