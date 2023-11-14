import numpy as np
import VAD_Alg


def lidar_vad_processing(lidar_data_object):
    """lidar_vad_processing takes in the lidar data object class and unpacks it into variables
    It then uses these to filter out NANs and compute the VAD extraction at each range ring
    with 3 or more non nan values"""
    # unpack object
    azimuth = lidar_data_object.azimuth
    elevation = lidar_data_object.elevation
    rad_vector = lidar_data_object.rad_vector
    ranger = lidar_data_object.range
    # create empty list for z output and vector output
    z_output = []
    output_vector = []

# loop over all ranges
    for i in range(np.size(azimuth, axis=0)):
        mask = ~np.isnan(rad_vector[i, :])  # mask that removed NaN values for a range ring
        if mask.sum() < 3:  # VAD cannot find solution with less than 3 points so pass
            pass
        else:
            # variable with masked values
            azimuth_mask = azimuth[i, mask]
            elevation_mask = elevation[i, mask]
            rad_vector_mask = rad_vector[i, mask]
            ranger_mask = ranger[i, mask]
            # cartesian components
            z = ranger_mask * np.sin(np.deg2rad(elevation_mask))
            x = ranger_mask * np.cos(np.deg2rad(elevation_mask)) * np.cos(np.deg2rad(azimuth_mask))
            y = ranger_mask * np.cos(np.deg2rad(elevation_mask)) * np.sin(np.deg2rad(azimuth_mask))
            # store a single z value as the average z value for ring
            z_output.append(np.average(z))
            # reshape vector for VAD
            radial_vector = np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1)), z.reshape((-1, 1))))
            # VAD on the dataset available
            VAD_output = VAD_Alg.vad_extraction(radial_vector, rad_vector_mask.reshape((-1, 1)))
            VAD_output_vector = VAD_output[0]
            output_vector.append(VAD_output_vector)

    #convert list to numpy array
    output_vector = np.stack(output_vector, axis=0)
    z_output = np.stack(z_output, axis=0)
    return output_vector.squeeze(), z_output

