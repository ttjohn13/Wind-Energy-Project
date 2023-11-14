import numpy as np


def Space_Field(Column_Vector, LengthWidth, number_of_points):
    column_size = np.size(Column_Vector)  # get size of column vector
    x = np.linspace(0, LengthWidth, number_of_points)
    y = np.linspace(0, LengthWidth, number_of_points)

    # repeat the column vector for all the space
    space = np.tile(Column_Vector, (number_of_points, number_of_points, 1, 1))
    return x, y, space


def kick(vector_field, max_kick):
    # create a matrix with random values between -1 and 1 then multiply by the percent of max kick to get percentage kick
    individual_change_percent = (2 * np.random.rand(*vector_field.shape)-1) * max_kick
    # mult percentage by the actual field to get the values of kick
    individual_change = np.multiply(vector_field, individual_change_percent)
    # add kick value to original vector field
    return vector_field + individual_change

