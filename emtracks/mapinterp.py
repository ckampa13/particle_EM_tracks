#! /usr/bin/env python

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def get_df_interp_func(filename=None, df=None, gauss=True, mm=False, scipy_interp=False, bounds=None, Blabels=['Bx','By','Bz']):
    '''
    This factory function will return an interpolating function for any field map. An input x,y,z will output the corresponding Bx,By,Bz or Br,Bphi,Bz. Will decide later if linear interpolation is good enough.

    Assumed file for input has length: meters, Bfield: Gauss
    '''
    # load dataframe if not passed in
    if df is None:
        if ".p" in filename:
            df = pd.read_pickle(filename)
        else:
            df = pd.read_csv(filename)
    else:
        df = df.copy()
    # labels
    Bx_l, By_l, Bz_l = Blabels
    if not gauss:
        df[Bx_l] = df[Bx_l] / 1e4
        df[By_l] = df[By_l] / 1e4
        df[Bz_l] = df[Bz_l] / 1e4
    if mm:
        df["X"] = df["X"] * 1e3
        df["Y"] = df["Y"] * 1e3
        df["Z"] = df["Z"] * 1e3

    xs = df.X.unique()
    ys = df.Y.unique()
    zs = df.Z.unique()

    if bounds is not None:
        xmin = xs[xs < bounds.xmin][-1]
        xmax = xs[xs > bounds.xmax][0]
        ymin = ys[ys < bounds.ymin][-1]
        ymax = ys[ys > bounds.ymax][0]
        zmin = zs[zs < bounds.zmin][-1]
        zmax = zs[zs > bounds.zmax][0]
        query_string = f"X>={xmin} & X<={xmax} & Y>={ymin} & Y<={ymax} & Z>={zmin} & Z<={zmax}"
        df = df.query(query_string)

        xs = df.X.unique()
        ys = df.Y.unique()
        zs = df.Z.unique()

    dx = xs[1]-xs[0]
    dy = ys[1]-ys[0]
    dz = zs[1]-zs[0]

    lx = len(xs)
    ly = len(ys)
    lz = len(zs)

    df_np = df[["X","Y","Z",Bx_l,By_l,Bz_l]].values

    x, y, z, bx, by, bz = df_np.T
    BX = bx.reshape(lx, ly, lz)
    BY = by.reshape(lx, ly, lz)
    BZ = bz.reshape(lx, ly, lz)

    interp_func_Bx = RegularGridInterpolator((xs, ys, zs), BX)
    interp_func_By = RegularGridInterpolator((xs, ys, zs), BY)
    interp_func_Bz = RegularGridInterpolator((xs, ys, zs), BZ)

    def scipy_interp_func(pts):
        Bx = interp_func_Bx(pts.T)
        By = interp_func_By(pts.T)
        Bz = interp_func_Bz(pts.T)
        return np.array([Bx, By, Bz])

    def get_cube(x, y, z):
        a_x, a_y, a_z = len(xs[xs <= x]) - 1, len(ys[ys <= y]) - 1, len(zs[zs <= z]) - 1
        corner_a = (ly * lz) * a_x + (lz) * a_y + a_z
        corner_b = corner_a + lz
        corner_c = corner_a + ly * lz
        corner_d = corner_a + ly * lz + lz
        index_list = [corner_a,corner_a+1,corner_b,corner_b+1,
        corner_c,corner_c+1,corner_d,corner_d+1]
        return df_np[index_list]

    def interp_single(xd,yd,zd,ff):
        c00 = ff[0,0,0]*(1 - xd) + ff[1,0,0] * xd
        c01 = ff[0,0,1]*(1 - xd) + ff[1,0,1] * xd
        c10 = ff[0,1,0]*(1 - xd) + ff[1,1,0] * xd
        c11 = ff[0,1,1]*(1 - xd) + ff[1,1,1] * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        return c0 * (1 - zd) + c1 * zd

    def interp(p_vec):
        cube = get_cube(*p_vec)

        xx = np.unique(cube[:,0])
        yy = np.unique(cube[:,1])
        zz = np.unique(cube[:,2])

        bxs_grid = cube[:,3].reshape((2,2,2))
        bys_grid = cube[:,4].reshape((2,2,2))
        bzs_grid = cube[:,5].reshape((2,2,2))

        xd = (p_vec[0]-xx[0])/(xx[1]-xx[0])
        yd = (p_vec[1]-yy[0])/(yy[1]-yy[0])
        zd = (p_vec[2]-zz[0])/(zz[1]-zz[0])

        bx = interp_single(xd,yd,zd, bxs_grid)
        by = interp_single(xd,yd,zd, bys_grid)
        bz = interp_single(xd,yd,zd, bzs_grid)

        return np.array([bx,by,bz])

    if scipy_interp:
        return scipy_interp_func
    else:
        return interp
