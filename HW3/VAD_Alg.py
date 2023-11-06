import numpy as np


def vad_extraction(radial_vectors, radial_velocity):
    vector_norm = np.linalg.norm(radial_vectors, axis=1, keepdims=True)
    normed_rad_vect = radial_vectors / vector_norm
    Xhat = np.linalg.lstsq(normed_rad_vect, radial_velocity)
    return Xhat
