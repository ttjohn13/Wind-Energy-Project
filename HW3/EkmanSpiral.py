import numpy as np
def Ekman_Spiral(Input_col_vector, y_location):
    y_top = y_location[-1] # top location
    relative_y = y_location/y_top # assigns each vertical point in terms of total height
    Output_col_vector = np.zeros_like(Input_col_vector) # creates the output column vector
    Output_col_vector[:, 1] = Input_col_vector[:, 0] * np.sin(np.pi/4 * (1-relative_y))  # creates a spiral with zero rotation up top and a max rotation of 45deg on bottom
    Output_col_vector[:, 0] = Input_col_vector[:, 0] * np.cos(np.pi/4 * (1-relative_y))  # creates a spiral with zero rotation up top and a max rotation of 45deg on bottom
    return Output_col_vector
