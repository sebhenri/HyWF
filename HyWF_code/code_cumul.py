from utils import *


# score functions
def score_accuracy_multi_class(ground_truth, predicted):
    ground_truth_mc = np.array(ground_truth).astype(int)
    predicted_mc = np.array(predicted).astype(int)
    ground_truth_tc = ground_truth_mc >= 0

    # true positive and false positive rates
    tpr = float(sum(np.logical_and(predicted_mc >= 0, np.equal(predicted_mc,ground_truth_mc))))/sum(ground_truth_tc)
    if sum(np.logical_not(ground_truth_tc)) > 0:
        fpr = float(sum(np.logical_and(predicted_mc >= 0, np.not_equal(predicted_mc,ground_truth_mc))))\
              /sum(np.logical_not(ground_truth_tc))
    else:
        fpr = 0
    score = tpr + (1-fpr)
    return score


def score_accuracy_two_class(ground_truth, predicted):
    ground_truth_tc = np.array([int(c) >= 0 for c in ground_truth])
    predicted_tc = np.array([int(c) >= 0 for c in predicted])
    # true positive and false positive
    tpr = float(sum(np.logical_and(predicted_tc, np.equal(predicted_tc,ground_truth_tc))))/sum(ground_truth_tc)
    if sum(np.logical_not(ground_truth_tc)) > 0:
        fpr = float(sum(np.logical_and(predicted_tc, np.not_equal(predicted_tc,ground_truth_tc))))\
              /sum(np.logical_not(ground_truth_tc))
    else:
        fpr = 0
    score = tpr + (1-fpr)
    return score


# Features for Panchenko et al. (NDSS 16): cumulative sum of packet sizes
def extract_Panchenko(times, sizes, num_features=104):
    if len(sizes) == 0:
        return num_features*[0]

    cumsum = np.cumsum(sizes)
    indices = np.linspace(0, len(sizes), num_features, endpoint=False)
    indices = list(map(int,indices))
    features = cumsum[indices]
    return features
