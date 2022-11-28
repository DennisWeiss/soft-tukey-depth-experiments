import csv
import os
import numpy as np


CLASS = 9
RESULT_NAME_DESC = 'var_max_2e-3_14epochs'
RUN = 1

result_path = f'results/CIFAR10_class{CLASS}_Autoencoder_{RESULT_NAME_DESC}_run{RUN}_kde/'

data0 = csv.reader(open(f'results/raw/soft_tukey_depths_CIFAR10_Autoencoder_Nominal_Encoder_{RESULT_NAME_DESC}_{CLASS}_run{RUN}.csv'), delimiter=',')
data1 = csv.reader(open(f'results/raw/soft_tukey_depths_CIFAR10_Autoencoder_Anomalous_Encoder_{RESULT_NAME_DESC}_{CLASS}_run{RUN}.csv'), delimiter=',')


tukey_depths = []


for data in [data0, data1]:
    for row, values in enumerate(data):
        print(row, len(values), data)
        if row == 0:
            tukey_depths.append(np.asarray(list(map(float, values))))

print(len(tukey_depths))

if not os.path.exists(result_path):
    os.makedirs(result_path)

