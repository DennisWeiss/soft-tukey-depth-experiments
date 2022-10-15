import scipy as sp
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
import sys

matplotlib.use('TkAgg')

data0 = csv.reader(open('soft_tukey_depths_CIFAR10_0.csv'), delimiter=',')
data1 = csv.reader(open('soft_tukey_depths_CIFAR10_1.csv'), delimiter=',')

tukey_depths = []

for data in [data0, data1]:
    for row, values in enumerate(data):
        if row == 0:
            tukey_depths.append(np.asarray(list(map(float, values))))

kde_0 = sp.stats.gaussian_kde(tukey_depths[0])
kde_1 = sp.stats.gaussian_kde(tukey_depths[1])
x = np.arange(0, 150, 0.02)
y0 = kde_0(x)
y1 = kde_1(x)

fig0_nominal = plt.figure()
plt.hist(tukey_depths[0][tukey_depths[0] < 50], bins=100)
fig0_nominal.savefig(sys.argv[1] + 'hist_nominal.png')

fig0_anomalous = plt.figure()
plt.hist(tukey_depths[1][tukey_depths[1] < 50], bins=100)
fig0_anomalous.savefig(sys.argv[1] + 'hist_anomalous.png')

fig1 = plt.figure()
plt.plot(x, y0, label='soft tukey depths of 0\'s')
plt.plot(x, y1, label='soft tukey depths of 1\'s')
plt.legend()
fig1.savefig(sys.argv[1] + 'kde.png')


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

for threshold in np.arange(0, 100, 0.2):
    true_positive_rates.append(get_true_positive_rate(tukey_depths[1], threshold))
    false_positive_rates.append(get_false_positive_rate(tukey_depths[0], threshold))

fig2 = plt.figure()
plt.plot(false_positive_rates, true_positive_rates)
fig2.savefig(sys.argv[1] + 'rate_curve.png')

print(compute_auroc(true_positive_rates, false_positive_rates))
