import numpy as np 
import pandas as pd 
import scipy.stats as st

act = [[2,4,5,7],[3,6,8,1]]
pred = [[1,2,3,4,5,6,7,8],
        [1,2,3,4,5,6,7,8]]

# recall@k function
def recall(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = round(len(act_set & pred_set) / float(len(act_set)), 2)
    # print(act_set)
    # print(pred_set)
    return result

actual = [2,4,5,7]
predicted = [1,2,3,4,5,6,7,8]
for k in range(1, 9):
    print(f"Recall@{k} = {recall(actual, predicted, k)}")
    
recall(actual, predicted, 3)    

class IR_metrics():
    def __init__(self, actual, predicted, k):
        self.actual = actual
        self.predicted = predicted
        self.k = k

    def recall_K2(self):
        act_set = set(self.actual)
        pred_set = set(self.predicted[:self.k])
        result = round(len(act_set & pred_set) / float(len(act_set)), 2)

        return result

    def map_K(self):
        if len(self.predicted)>self.k:
            pred = self.predicted[:self.k]
        else :
            pred = self.predicted
        score = 0.0
        num_hits = 0.0

        for i,p in enumerate(pred):
            if p in self.actual and p not in pred[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)

        if not self.actual:
            return 0.0

        return score / min(len(self.actual), self.k)

    def precision_K(self):
        pred = self.predicted[:self.k]
        num_hit = len(set(pred).intersection(set(self.actual)))
        precision = float(num_hit) / len(pred)
        return precision

    def recall_K(self):
        pred = self.predicted[:self.k]
        num_hit = len(set(pred).intersection(set(self.actual)))
        recall = float(num_hit) / len(self.actual)
        return recall    

    def F1_K(self):
        pred = self.predicted[:self.k]
        num_hit = len(set(pred).intersection(set(self.actual)))
        precision = float(num_hit) / len(pred)
        recall = float(num_hit) / len(self.actual)
        f1  = 2*(precision)*(recall) / (precision+recall)
        return f1

    def mrr_K(self):
        # self.predicted[0]
        print('mrr_K:',self.predicted[:self.k][0])
        x = self.predicted[:self.k][0]
        rank = (1/(x+1)) ### Because of the zero division error 1 added to denominator
        
        return rank


for i in range(len(act)):
    k=8
    print(IR_metrics(act[i], pred[i], 5).recall_K2())
    print('map@'+str(k),IR_metrics(act[i], pred[i], k).map_K()) ## bu kisim eksik sounc append edilip mean degerinin alinmasi gerek 
    print('Precision@'+str(k),IR_metrics(act[i], pred[i], k).precision_K())
    print('recall@'+str(k),IR_metrics(act[i], pred[i], k).recall_K()) ### check this one also
    print('F1@'+str(k),IR_metrics(act[i], pred[i], k).F1_K())
    print('mrr@'+str(k),IR_metrics(act[i], pred[i], k).mrr_K())
    
    
    
    

# def ndcg(rank, k):
#     """Normalized Discounted Cumulative Gain.
#     Args:
#         :param rank: A list.
#         :param k: A scalar(int).
#     :return: ndcg.
#     """
#     res = 0.0
#     for r in rank:
#         if r < k:
#             res += 1 / np.log2(r + 2)
#     return res / len(rank)
# arr = np.array([4,3,1,8,7,19,11])
# K = 5

# ndcg(arr,K)
# https://github.com/ZiyaoGeng/RecLearn/blob/reclearn/reclearn/evaluator/metrics.py
