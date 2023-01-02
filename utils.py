import numpy as np



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


def get_auroc(nominal_tukey_depths, anomalous_tukey_depths):
    true_positive_rates = []
    false_positive_rates = []

    for threshold in np.arange(0, 0.5, 1e-5):
        true_positive_rates.append(get_true_positive_rate(anomalous_tukey_depths, threshold))
        false_positive_rates.append(get_false_positive_rate(nominal_tukey_depths, threshold))

    return compute_auroc(true_positive_rates, false_positive_rates)