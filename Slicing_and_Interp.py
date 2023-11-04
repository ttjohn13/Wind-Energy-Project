import numpy as np
import scipy as sp


def rad2CartOffset(az, r, elv):
    xoffset =

def slice_points(data, x, y, z, az, rang, elev, number_of_measurements=100, lidar_loc=np.NAN):
    udata = data[:, :, :, 0]
    vdata = data[:, :, :, 1]

    uinterp = sp.interpolate.RegularGridInterpolator((x, y, z,), udata)
    vinterp = sp.interpolate.RegularGridInterpolator((x, y, z), vdata)

    if np.isnan(lidar_loc):
        zloc = 0
        xloc = max(x) / 2
        yloc = max(y) / 2
        lidar_loc = np.array([xloc, yloc, zloc])

