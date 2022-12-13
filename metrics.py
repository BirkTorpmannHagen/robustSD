
def fprat95tpr(p_values, labels):
    """
    :param p_values: p-values as returned from the tests at a given sample size
    :param labels: labels describing whether the samples are ood or ind
    Find p-value threshold that results in 95% TPR. Then find FPR.
    :return:
    """
    pass

def detection_error():
    pass

def auroc():
    pass

def aupr():
    pass

def sample_sensitivity():
    """
    Quantifies the sensitivity of a given method to sample size.
    Essentially: area under the detection-error v sample size curve.
    :return:
    """

def correlation(p_values, performance_drops):
    """

    :param p_values: p-values as returned from the tests at a given sample size
    :param performance_drops: percentage drops relative to InD, as quantified by the loss value
    :return:
    """