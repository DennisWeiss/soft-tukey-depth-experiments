import scipy as sp
import scipy.stats
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
import sys
import os


CLASS = 0

result_path = f'results/MNIST_class{CLASS}_Encoder/'

data0 = csv.reader(open(f'results/raw/soft_tukey_depths_MNIST_Autoencoder_NominalMNISTAutoencoderDataset_{CLASS}.csv'), delimiter=',')
data1 = csv.reader(open(f'results/raw/soft_tukey_depths_MNIST_Autoencoder_AnomalousMNISTAutoencoderDataset_{CLASS}.csv'), delimiter=',')

tukey_depths = []


for data in [data0, data1]:
    for row, values in enumerate(data):
        print(row, len(values), data)
        if row == 0:
            tukey_depths.append(np.asarray(list(map(float, values))))

print(len(tukey_depths))

if not os.path.exists(result_path):
    os.makedirs(result_path)


kde_0 = sp.stats.gaussian_kde(tukey_depths[0], bw_method=1e-1)
kde_1 = sp.stats.gaussian_kde(tukey_depths[1], bw_method=1e-1)
x = np.arange(0, 0.5, 1e-5)
y0 = kde_0(x)
y1 = kde_1(x)

fig0_nominal = plt.figure()
plt.hist(tukey_depths[0][tukey_depths[0] < 0.5], bins=50)
plt.xlabel('soft Tukey depth')
plt.ylabel('count')
plt.title(f'Histogram of soft Tukey depths of test {CLASS} class w.r.t. train {CLASS} class')
fig0_nominal.savefig(result_path + 'hist_nominal.png')

fig0_anomalous = plt.figure()
plt.hist(tukey_depths[1][tukey_depths[1] < 0.5], bins=50, color='orange')
plt.xlabel('soft Tukey depth')
plt.ylabel('count')
plt.title(f'Histogram of soft Tukey depths of test non-{CLASS} classes w.r.t. train {CLASS} class')
fig0_anomalous.savefig(result_path + 'hist_anomalous.png')

fig1 = plt.figure()
plt.plot(x, y0, label=f'soft Tukey depths of test {CLASS} class')
plt.plot(x, y1, label=f'soft Tukey depths of test non-{CLASS} classes')
plt.title(f'KDE of soft Tukey depths of test {CLASS} class and non-{CLASS} classes w.r.t. train {CLASS} class')
plt.xlabel('soft Tukey depth')
plt.ylabel('p')
plt.legend()
fig1.savefig(result_path + 'kde.png')


def get_true_positive_rate(anomalous_tukey_depths, threshold):
    true_positives = 0
    for anomalous_tukey_depth in anomalous_tukey_depths:
        if anomalous_tukey_depth < threshold:
            true_positives += 1
    return true_positives / len(anomalous_tukey_depths)


def get_false_positive_rate(nominal_tukey_depths, threshold):
    false_positive = 0
    for nominal_tukey_depth in nominal_tukey_depths:
        if nominal_tukey_depth < threshold:
            false_positive += 1
    return false_positive / len(nominal_tukey_depths)


def compute_auroc(true_positive_rates, false_positive_rates):
    auroc = 0
    for i in range(len(true_positive_rates)):
        if i == 0:
            auroc += (1 - 0.5 * false_positive_rates[i]) * true_positive_rates[i]
        else:
            auroc += (1 - 0.5 * false_positive_rates[i] - 0.5 * false_positive_rates[i-1]) * (true_positive_rates[i] - true_positive_rates[i-1])
    return auroc


true_positive_rates = []
false_positive_rates = []

for threshold in np.arange(0, 0.5, 3e-6):
    true_positive_rates.append(get_true_positive_rate(tukey_depths[1], threshold))
    false_positive_rates.append(get_false_positive_rate(tukey_depths[0], threshold))

fig2 = plt.figure()
plt.plot(false_positive_rates, true_positive_rates)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
fig2.savefig(result_path + 'rate_curve.png')


auroc = compute_auroc(true_positive_rates, false_positive_rates)

print(f'AUROC: {auroc}')

result_file = open(result_path + 'result.txt', 'w')

result_file.write(f'AUROC: {auroc}')
