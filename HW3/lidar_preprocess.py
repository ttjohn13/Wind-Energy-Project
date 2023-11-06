import numpy as np
import VAD_Alg


def lidar_vad_processing(lidar_data_object):
    azimuth = lidar_data_object.azimuth
    elevation = lidar_data_object.elevation
    rad_vector = lidar_data_object.rad_vector
    ranger = lidar_data_object.range
    z_output = []
    output_vector = []

    for i in range(np.size(azimuth, axis=0)):
        mask = ~np.isnan(rad_vector[i, :])
        if mask.sum() < 3:
            pass
        else:
            azimuth_mask = azimuth[i, mask]
            elevation_mask = elevation[i, mask]
            rad_vector_mask = rad_vector[i, mask]
            ranger_mask = ranger[i, mask]
            z = ranger_mask * np.sin(np.deg2rad(elevation_mask))
            x = ranger_mask * np.cos(np.deg2rad(elevation_mask)) * np.cos(np.deg2rad(azimuth_mask))
            y = ranger_mask * np.cos(np.deg2rad(elevation_mask)) * np.sin(np.deg2rad(azimuth_mask))
            z_output.append(np.average(z))
            radial_vector = np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1)), z.reshape((-1, 1))))
            VAD_output = VAD_Alg.vad_extraction(radial_vector, rad_vector_mask.reshape((-1, 1)))
            VAD_output_vector = VAD_output[0]
            output_vector.append(VAD_output_vector)

    output_vector = np.stack(output_vector, axis=0)
    z_output = np.stack(z_output, axis=0)
    return output_vector.squeeze(), z_output

