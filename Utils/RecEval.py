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
def Precision_and_Recall(ranklist, itemdict):
    rankinglist = np.asarray(ranklist, dtype=np.int32).flatten()
    testinglist = np.asarray(list(itemdict.keys()), dtype=np.int32).flatten()

    sum_relevant_item = 0
    for item in testinglist:
        if item in rankinglist:
            sum_relevant_item += 1

    precision = sum_relevant_item / len(rankinglist)
    recall = sum_relevant_item / len(testinglist)
    return precision, recall

def Recall(ranklist, itemdict):
    rankinglist = np.asarray(ranklist, dtype=np.int32).flatten()
    testinglist = np.asarray(list(itemdict.keys()), dtype=np.int32).flatten()

    sum_relevant_item = 0
    for item in testinglist:
        if item in rankinglist:
            sum_relevant_item += 1

    return sum_relevant_item / len(testinglist)

def AP(ranklist, itemdict):
    rankinglist = np.asarray(ranklist, dtype=np.int32).flatten()
    testinglist = np.asarray(list(itemdict.keys()), dtype=np.int32).flatten()

    precision, rel = [], []
    for k in range(len(rankinglist)): # Loop the ranking list and calculate each precision for each loop

        if rankinglist[k] in testinglist: # The k-th item is relevant
            rel.append(1)
        else:
            rel.append(0)

        sum_relevant_item = 0       # Precision up-to k-th item
        rkl = rankinglist[:k+1]
        for item in testinglist:
            if item in rkl:
                sum_relevant_item += 1
        precision.append(sum_relevant_item / len(rkl))

    return np.sum(np.asarray(precision) * np.asarray(rel)) / min(len(rankinglist), len(testinglist))

def HR(ranklist, itemdict):
    # rankinglist = np.asarray(ranklist, dtype=np.int32).flatten()
    # testinglist = np.asarray(list(itemdict.keys()), dtype=np.int32).flatten()

    # assert len(testinglist) == 1 # In calculating hit rate, the length of the item list must be 1
    item = list(itemdict.keys())[0]
    if item in ranklist:
        return 1
    else:
        return 0

def NDCG_One(ranklist, itemdict):
    item = list(itemdict.keys())[0]
    if item in ranklist:
        idx = np.argwhere(ranklist == item).item()
        return 1.0 / np.log2(idx+2)
    else:
        return 0.0

def NDCG_Full(ranklist, itemdict):
    rankinglist = np.asarray(ranklist,dtype=np.int32).flatten()
    testinglist = np.asarray(list(itemdict.keys()),dtype=np.int32).flatten()

    dcg_sum = 0.0
    for item in testinglist:
        if item in rankinglist:
            idx = np.argwhere(rankinglist == item).item()
            dcg_sum += itemdict[item] / np.log2(idx+2) # Get the DCG sum

    idcg_sum = 0.0
    # Ideally, all items are ranked at the head of the resulting list
    ideal_ranking = sorted(itemdict.items(),key=lambda x: x[1], reverse=True)
    for pos, (item, rating) in enumerate(ideal_ranking):
        idcg_sum += rating / np.log2(pos+2)

    return dcg_sum / idcg_sum

def MRR_One(ranklist, itemdict):
    item = list(itemdict.keys())[0]
    if item in ranklist:
        idx = np.argwhere(ranklist == item).item()
        return 1.0 / (idx+1)
    else:
        return 0.0

def MRR_Full(ranklist, itemdict):
    rankinglist = np.asarray(ranklist).flatten()
    testinglist = np.asarray(list(itemdict.keys()), dtype=np.int32).flatten()

    sum_rr = 0.0
    for item in testinglist:
        if item in rankinglist:
            idx = np.argwhere(rankinglist == item).item()
            sum_rr += 1.0/(idx+1)

    return sum_rr / len(testinglist)

# Calculate all relevant ranking metrics
def rankingMetrics(scores, itemlist, K_list, test_itemdict, mod = 'hr', is_map=False, is_mrr=False, is_ndcg=False):
    assert len(scores) == len(itemlist)
    # Construct the score list
    scoredict = {}
    for i in range(len(scores)):
        scoredict[itemlist[i]] = scores[i]

    # Get the top K scored items
    ranklist = heapq.nlargest(max(K_list), scoredict, key=scoredict.get)

    if mod == 'hr':
        hr_list, ndcg_list = [], []
        for k in K_list:
            hr_list.append(HR(ranklist[:k], test_itemdict))
            ndcg_list.append(NDCG_One(np.asarray(ranklist[:k]), test_itemdict))

        return hr_list, ndcg_list

    if mod == 'precision':
        prec_list, recall_list, ap_list, mrr_list, ndcg_list = [], [], [], [], []
        for k in K_list:
            p, r = Precision_and_Recall(ranklist[:k], test_itemdict)
            prec_list.append(p)
            recall_list.append(r)
            if is_map:
                ap = AP(ranklist[:k], test_itemdict)
                ap_list.append(ap)
            if is_mrr:
                mrr = MRR_Full(ranklist[:k], test_itemdict)
                mrr_list.append(mrr)
            if is_ndcg:
                ndcg = NDCG_Full(ranklist[:k], test_itemdict)
                ndcg_list.append(ndcg)

        return prec_list, recall_list, ap_list, mrr_list, ndcg_list


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