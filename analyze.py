from time import sleep

import torch
import torch.utils.data
import torchvision
import numpy as np
import csv
import matplotlib.pyplot as plt

from DataLoader import NominalMNISTImageDataset, AnomalousMNISTImageDataset

CLASS = 0
RESULT_NAME_DESC = 'temp2'
RUN = 0

result_path = f'results/MNIST_Autoencoder_class{CLASS}_{RESULT_NAME_DESC}_run{RUN}/'

data0 = csv.reader(open(f'results/raw/soft_tukey_depths_MNIST_Autoencoder_Nominal_{RESULT_NAME_DESC}_{CLASS}.csv'), delimiter=',')
data1 = csv.reader(open(f'results/raw/soft_tukey_depths_MNIST_Autoencoder_Anomalous_{RESULT_NAME_DESC}_{CLASS}.csv'), delimiter=',')


for type, data in [['Nominal', data0], ['Anomalous', data1]]:
    for row, values in enumerate(data):
        if len(values) > 0:
            test_data = torch.utils.data.Subset(
                (NominalMNISTImageDataset if type == 'Nominal' else AnomalousMNISTImageDataset)(nominal_class=CLASS, train=False),
                list(range(len(values)))
            )
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=len(values))
            for step, X in enumerate(test_dataloader):
                print(row, len(values), data)
                td_values = [(i, float(x)) for i, x in enumerate(values)]
                td_values.sort(key=lambda x: x[1])
                print(td_values)
                for i in range(8):
                    plt.imshow(np.transpose(X[td_values[i][0]].numpy(), (1, 2, 0)))
                    plt.show()
                    sleep(0.1)
                for i in range(8):
                    plt.imshow(np.transpose(X[td_values[len(values)-1-i][0]].numpy(), (1, 2, 0)))
                    plt.show()
                    sleep(0.1)


