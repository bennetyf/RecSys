'''
This module is implementations of common evaluation metrics in recommender systems
'''

import numpy as np
import heapq
import sklearn.metrics as metrics

############# Rating Prediction Metrics

def evalMAE(ground_truth, predictions):
    # Flatten the input array-like data into 1-D
    truth = np.asarray(ground_truth).flatten()
    pred = np.asarray(predictions).flatten()
    return metrics.mean_absolute_error(y_true=truth, y_pred=pred)

def evalRMS(ground_truth, predictions):
    # Flatten the input array-like data into 1-D
    truth = np.asarray(ground_truth).flatten()
    pred = np.asarray(predictions).flatten()
    return metrics.mean_squared_error(y_true=truth, y_pred=pred)

############# Top-K Ranking Metrics

def evalHR(ranklist, item):
    res = np.asarray(ranklist).flatten()
    if item in res:
        return 1
    else:
        return 0

def evalNDCG(ranklist, item):
    res = np.asarray(ranklist).flatten()

    # Caculate nDCG value
    # (Only one item is relevant in the list)
    # All the other relevance values are set to zero
    # Only the position of the target item matters
    if item in res:
        idx = np.argwhere(res == item).item()
        return 1.0 / np.log2(idx+2)
    else:
        return 0.0

def evalRR(ranklist, item):
    res = np.asarray(ranklist).flatten()
    if item in res:
        idx = np.argwhere(res == item).item()
        return 1.0 / (idx+1)
    else:
        return np.inf


def evalTopK(scores, itemlist, K, itempos = None):
    assert len(scores) ==  len(itemlist)

    if itempos == None:
        itempos = len(itemlist)-1

    # Get the target item from the original item list
    target_item = itemlist[itempos]

    # Construct the score list
    scoredict = {}
    for i in range(len(scores)):
        scoredict[itemlist[i]] = scores[i]

    # Get the top K scored items
    ranklist = heapq.nlargest(K, scoredict, key=scoredict.get)

    # Return the metrics
    return ranklist,evalHR(ranklist,target_item),evalNDCG(ranklist,target_item),evalRR(ranklist,target_item)

####################################################################################################################

# if __name__ == "__main__":
    # score = [4,2,3,1,7,4]
    # itemlist =[1234,5678,1111,2222,3333,4444]
    # print(evalTopK(score, itemlist, 3))