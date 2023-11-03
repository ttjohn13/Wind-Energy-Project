import numpy as np
import scipy.io as sio

mat_contents = sio.loadmat('Data_for_VAD.mat')  # import matlab file
Data = mat_contents['Data']  # pull the data structure
Datestring = Data[:, :]['name']  # pull the name profile that defines date and time

import datetime


def convert_to_datetime(date_string):  # Function to convert the string of date times to a date time format
    return datetime.datetime.strptime(date_string, "%Y%m%d_%H%M%S")


# conduct conversion function in an inline for loop
Dates = np.array([convert_to_datetime(Datestring[0, i][0]) for i in range(len(Data['name'][0]))])
number_of_arrays = np.shape(Data['range'][0, :])[0]  # count the number of scans


# Class will allow for my own defined organization of the matlab matrix
class LidarData:
    def __init__(self, ranges, az, el, rv, date):
        self.range = ranges
        self.azimuth = az
        self.elevation = el
        self.rad_vector = rv
        self.date = date


Formatted_data = []  # create an empty list to store the new class objects with data

# Fill list with the object files
for i in range(number_of_arrays):
    range_array = Data['range'][0, i][:, :]
    az_array = Data['az'][0, i][:, :]
    el_array = Data['el'][0, i][:, :]
    rv_array = Data['rv'][0, i][:, :]
    Formatted_data.append(LidarData(range_array, az_array, el_array, rv_array, Dates[i]))


import pickle

with open('Formatted_Data.pickle', 'wb') as f:
    pickle.dump(Formatted_data, f)
