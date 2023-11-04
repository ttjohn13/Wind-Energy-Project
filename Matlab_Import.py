import numpy as np
import scipy.io as sio
import datetime


class LidarData:
    def __init__(self, ranges, az, el, rv, date):
        self.range = ranges
        self.azimuth = az
        self.elevation = el
        self.rad_vector = rv
        self.date = date


def convert_to_datetime(date_string):  # Function to convert the string of date times to a date time format
    return datetime.datetime.strptime(date_string, "%Y%m%d_%H%M%S")


def matlabimport():
    mat_contents = sio.loadmat('Data_for_VAD.mat')  # import matlab file
    data = mat_contents['Data']  # pull the data structure
    datestring = data[:, :]['name']  # pull the name profile that defines date and time

    # conduct conversion function in an inline for loop
    dates = np.array([convert_to_datetime(datestring[0, i][0]) for i in range(len(data['name'][0]))])
    number_of_arrays = np.shape(data['range'][0, :])[0]  # count the number of scans
    # Class will allow for my own defined organization of the matlab matrix
    formatted_data = []  # create an empty list to store the new class objects with data
    # Fill list with the object files
    for i in range(number_of_arrays):
        range_array = data['range'][0, i][:, :]
        az_array = data['az'][0, i][:, :]
        el_array = data['el'][0, i][:, :]
        rv_array = data['rv'][0, i][:, :]
        formatted_data.append(LidarData(range_array, az_array, el_array, rv_array, dates[i]))

    return formatted_data
