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
    z_offset = np.resize(z_offset, (len(x_offset), 1))
    cartesian_offset = np.hstack((x_offset, y_offset, z_offset))
    return cartesian_offset


def slice_points(data, x, y, z, az, elev, zplane, lidar_loc=np.NAN):
    udata = data[:, :, :, 0]
    vdata = data[:, :, :, 1]
    rang = np.divide(zplane, np.sin(np.deg2rad(elev)))

    uinterp = sp.interpolate.RegularGridInterpolator((x, y, z), udata)
    vinterp = sp.interpolate.RegularGridInterpolator((x, y, z), vdata)

    if np.isnan(lidar_loc).any():
        zloc = 0
        xloc = max(x) / 2
        yloc = max(y) / 2
        lidar_loc = np.array([xloc, yloc, zloc])

    offsets = rad2CartOffset(az, rang, elev)

    locations = lidar_loc + offsets
    u_interp_val = uinterp(locations)
    v_interp_val = vinterp(locations)
    x_return = locations[:, 0]
    y_return = locations[:, 1]
    z_return = locations[:, 2]
    return u_interp_val, v_interp_val, x_return, y_return, z_return


def vectorProjection(u_actual, v_actual, meas_location, lidar_location):
    laser_vector = meas_location - lidar_location.T
    laser_meas = np.zeros_like(u_actual)
    for i in range(len(u_actual)):
        wind_vector = np.array([[u_actual[i], v_actual[i], 0]])
        laser_meas[i] = np.dot(laser_vector[i, :].reshape((1, -1)), wind_vector.reshape((-1, 1))) / np.linalg.norm(laser_vector[i, :].reshape((-1, 1)))

    return laser_vector, laser_meas
