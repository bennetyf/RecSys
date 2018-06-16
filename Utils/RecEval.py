'''
This module is implementations of common evaluation metrics in recommender systems
'''

import numpy as np
import heapq
import sklearn.metrics as metrics

############# Rating Prediction Metrics
def MAE(ground_truth, predictions):
    # Flatten the input array-like data into 1-D
    truth = np.asarray(ground_truth).flatten()
    pred = np.asarray(predictions).flatten()
    return metrics.mean_absolute_error(y_true=truth, y_pred=pred)

def RMS(ground_truth, predictions):
    # Flatten the input array-like data into 1-D
    truth = np.asarray(ground_truth).flatten()
    pred = np.asarray(predictions).flatten()
    return np.sqrt(metrics.mean_squared_error(y_true=truth, y_pred=pred))

# Calculate all relevant prediction metrics
def predictionMetrics(labels, predictions):
    return MAE(labels, predictions), RMS(labels, predictions)

############# Top-K Ranking Metrics
def HR(ranklist, itemdict):
    rkl = np.asarray(ranklist, dtype=np.int32).flatten()
    itl = np.asarray(list(itemdict.keys()), dtype=np.int32).flatten()

    sum_hit = 0.0
    for item in itl:
        if item in rkl:
            sum_hit += 1

    return sum_hit / len(itl)

def NDCG(ranklist, itemdict):
    rkl = np.asarray(ranklist,dtype=np.int32).flatten()
    itl = np.asarray(list(itemdict.keys()),dtype=np.int32).flatten()

    dcg_sum = 0.0
    for item in itl:
        if item in rkl:
            idx = np.argwhere(rkl == item).item()
            dcg_sum += itemdict[item] / np.log2(idx+2) # Get the DCG sum

    idcg_sum, pos = 0.0, 0.0
    # Ideally, all items are ranked at the head of the resulting list
    ideal_ranking = sorted(itemdict.items(),key=lambda x: x[1], reverse=True)
    for item, rating in ideal_ranking:
        idcg_sum += rating / np.log2(pos+2)
        pos += 1

    return dcg_sum / idcg_sum

def MRR(ranklist, itemdict):
    rkl = np.asarray(ranklist).flatten()
    itl = np.asarray(list(itemdict.keys()), dtype=np.int32).flatten()

    sum_rr = 0.0
    for item in itl:
        if item in rkl:
            idx = np.argwhere(rkl == item).item()
            sum_rr += 1.0/(idx+1)

    return sum_rr / len(itl)

# Calculate all relevant ranking metrics
def rankingMetrics(scores, itemlist, K, test_itemdict):
    assert len(scores) == len(itemlist)
    # Construct the score list
    scoredict = {}
    for i in range(len(scores)):
        scoredict[itemlist[i]] = scores[i]

    # Get the top K scored items
    ranklist = heapq.nlargest(K, scoredict, key=scoredict.get)

    return HR(ranklist, test_itemdict), NDCG(ranklist,test_itemdict), MRR(ranklist,test_itemdict)

############# Top-K Ranking Metrics (Old)

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

def evalMRR(ranklist, item):
    res = np.asarray(ranklist).flatten()
    if item in res:
        idx = np.argwhere(res == item).item()
        return 1.0 / (idx+1)
    else:
        return 0.0

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
    return ranklist,evalHR(ranklist,target_item),evalNDCG(ranklist,target_item),evalMRR(ranklist,target_item)

####################################################################################################################

# if __name__ == "__main__":
    # score = [4,2,3,1,7,4]
    # itemlist =[1234,5678,1111,2222,3333,4444]
    # print(evalTopK(score, itemlist, 3))