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
    # calculate cartesian components
    z_offset = np.multiply(np.sin(np.deg2rad(elv)), r)
    x_offset = np.multiply(np.multiply(np.cos(np.deg2rad(elv)), np.cos(np.deg2rad(az))), r)
    y_offset = np.multiply(np.multiply(np.cos(np.deg2rad(elv)), np.sin(np.deg2rad(az))), r)
    z_offset = np.resize(z_offset, (len(x_offset), 1))  # copies single z value into vector of same size as x and y
    cartesian_offset = np.hstack((x_offset, y_offset, z_offset))
    return cartesian_offset


def slice_points(data, x, y, z, az, elev, zplane, lidar_loc=np.NAN):
    """ computes range ring values from azimuth elevation and height parameters
    :param data: wind speed data for whole space, both u and v
    :param x: x grid space locations, a single column of data
    :param y: y grid space locations, a single column of data
    :param z: z grid space locations, a single column of data
    :param az: column vector of azimuthal angles
    :param elev: a single elevation value in a 1x1 matrix
    :param zplane: height for measurement, should be 1 value 1x1 matrix
    :param lidar_loc: cartesian location of lidar machine. Default will be at halfway point for x and y space and on the ground
    :returns: the u and v interpolated vectors along ring, and the cartesian position of each ring point. 5 different ouputs
    """
    # grab u and v data separately
    udata = data[:, :, :, 0]
    vdata = data[:, :, :, 1]
    rang = np.divide(zplane, np.sin(np.deg2rad(elev)))  # create range values that match with z planes

    # create interpolation object that will be used for u and v
    uinterp = sp.interpolate.RegularGridInterpolator((x, y, z), udata)
    vinterp = sp.interpolate.RegularGridInterpolator((x, y, z), vdata)

    # center of bottom plane for a default lidar location
    if np.isnan(lidar_loc).any():
        zloc = 0
        xloc = max(x) / 2
        yloc = max(y) / 2
        lidar_loc = np.array([xloc, yloc, zloc])

    # calculate offsets
    offsets = rad2CartOffset(az, rang, elev)
    # location of each point on range ring
    locations = lidar_loc + offsets
    # interpolation
    u_interp_val = uinterp(locations)
    v_interp_val = vinterp(locations)
    x_return = locations[:, 0]
    y_return = locations[:, 1]
    z_return = locations[:, 2]
    return u_interp_val, v_interp_val, x_return, y_return, z_return


def vectorProjection(u_actual, v_actual, meas_location, lidar_location):
    """ projects vectors in direction of laser for a range ring
    :param u_actual: column vectors of u vectors on ring
    :param v_actual: column vectors of v vectors on ring
    :param meas_location: nx3 matrix of cartesian measurement locations
    :param lidar_location: 3 point cartesian location of lidar
    :returns: 2 variables - laser vector and the speed"""
    # find vector from lidar to location
    laser_vector = meas_location - lidar_location.T
    laser_meas = np.zeros_like(u_actual)  # space for speed measurements
    # loop that finds laser measurements
    for i in range(len(u_actual)):
        wind_vector = np.array([[u_actual[i], v_actual[i], 0]])
        laser_meas[i] = np.dot(laser_vector[i, :].reshape((1, -1)), wind_vector.reshape((-1, 1))) / np.linalg.norm(laser_vector[i, :].reshape((-1, 1)))

    return laser_vector, laser_meas
