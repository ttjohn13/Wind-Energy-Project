import numpy as np


def Space_Field(Column_Vector, LengthWidth, number_of_points):
    column_size = np.size(Column_Vector)  # get size of column vector
    x = np.linspace(0, LengthWidth, number_of_points)
    y = np.linspace(0, LengthWidth, number_of_points)

    space = np.tile(Column_Vector, (number_of_points, number_of_points, 1, 1))
    return x, y, space


def kick(vector_field, max_kick):
    individual_change_percent = (2 * np.random.rand(*vector_field.shape)-1) * max_kick
    individual_change = np.multiply(vector_field, individual_change_percent)
    return vector_field + individual_change

