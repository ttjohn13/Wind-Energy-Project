import numpy as np
import scipy as sp


def rad2CartOffset(az, r, elv):
    """
    Calculate cartesian offset from measuring point
    :param az: nx1 matrix of azimuthal degree angles
    :param r: nx1 matrix of radial distances
    :param elv: nx1 matrix of elevation angles from ground plane
    :return: nx3 matrix of cartesian offsets
    """
    z_offset = np.multiply(np.sin(np.deg2rad(elv)), r)
    x_offset = np.multiply(np.multiply(np.cos(np.deg2rad(elv)), np.cos(np.deg2rad(az))), r)
    y_offset = np.multiply(np.multiply(np.cos(np.deg2rad(elv)), np.sin(np.deg2rad(az))), r)
    cartesian_offset = np.hstack((x_offset, y_offset, z_offset))
    return cartesian_offset


def slice_points(data, x, y, z, az, elev, lidar_loc=np.NAN):
    udata = data[:, :, :, 0]
    vdata = data[:, :, :, 1]
    rang = np.divide(z, np.sin(np.deg2rad(elev)))

    uinterp = sp.interpolate.RegularGridInterpolator((x, y, z), udata)
    vinterp = sp.interpolate.RegularGridInterpolator((x, y, z), vdata)

    if np.isnan(lidar_loc):
        zloc = 0
        xloc = max(x) / 2
        yloc = max(y) / 2
        lidar_loc = np.array([xloc, yloc, zloc])

    offsets = np.zeros((len(rang), len(az), 3))
    for i in range(len(rang)):
        offsets[i, :, :] = rad2CartOffset(az, rang[i], elev)

    locations = lidar_loc + offsets
    u_interp_val = uinterp(locations)
    v_interp_val = vinterp(locations)
    x_return = locations[:, 0]
    y_return = locations[:, 1]
    z_return = locations[:, 2]
    return u_interp_val, v_interp_val, x_return, y_return, z_return
