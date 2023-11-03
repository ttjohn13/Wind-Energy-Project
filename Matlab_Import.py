import numpy as np
import scipy.io as sio
from os.path import dirname, join as pjoin
#data_dir = pjoindirname(MAE579)
#mat_fname = pjoin(data_dir, 'Data_for_VAD.mat')
mat_contents = sio.loadmat('Data_for_VAD.mat')
Data = mat_contents['Data']
Datestring = Data[:, :]['name']

import datetime
def conver_to_datetime(date_string):
    return datetime.datetime.strptime(date_string, "%Y%m%d_%H%M%S")




Dates = np.array([conver_to_datetime(Datestring[0, i][0]) for i in range(len(Data['name'][0]))])
number_of_arrays = np.shape(Data['range'][0, :])[0]
length_array, width_array = np.shape(Data['range'][0, 0])
# allocate memory for each set of arrays
range_array = np.zeros((number_of_arrays, length_array, width_array))
az_array = np.zeros_like(range_array)
el_array = np.zeros_like(range_array)
rv_array = np.zeros_like(range_array)

for i in range(number_of_arrays):
    range_array[i, :, :] = Data['range'][0, i][:, :]
    az_array[i, :, :] = Data['az'][0, i][:, :]
    el_array[i, :, :] = Data['el'][0, i][:, :]
    rv_array[i, :, :] = Data['rv'][0, i][:, :]



