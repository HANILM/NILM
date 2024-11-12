from nilmtk.electric import align_two_meters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
def tp_tn_fp_fn(states_pred, states_ground):
    tp = np.sum(np.logical_and(states_pred == 1, states_ground == 1))
    fp = np.sum(np.logical_and(states_pred == 1, states_ground == 0))
    fn = np.sum(np.logical_and(states_pred == 0, states_ground == 1))
    tn = np.sum(np.logical_and(states_pred == 0, states_ground == 0))
    return tp, tn, fp, fn

def recall_precision_accuracy_f1(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    threshold = ground.on_power_threshold()
    chunk_results = []
    sum_samples = 0.0
    for chunk in aligned_meters:
        sum_samples += len(chunk[5400:10800])
        pr = np.array([0 if (p)<threshold else 1 for p in chunk[5400:10800].iloc[:,0]])
        gr = np.array([0 if p<threshold else 1 for p in chunk[5400:10800].iloc[:,1]])

        tp, tn, fp, fn = tp_tn_fp_fn(pr,gr)
        p = sum(pr)
        n = len(pr) - p

        chunk_results.append([tp,tn,fp,fn,p,n])

    if sum_samples == 0:
        return None
    else:
        [tp,tn,fp,fn,p,n] = np.sum(chunk_results, axis=0)

        res_recall = recall(tp,fn)
        res_precision = precision(tp,fp)
        res_f1 = f1(res_precision,res_recall)
        res_accuracy = accuracy(tp,tn,p,n)

        return (res_recall,res_precision,res_accuracy,res_f1)

def relative_error_total_energy(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    chunk_results = []
    sum_samples = 0.0
    for chunk in aligned_meters:
        #chunk.fillna(0, inplace=True)
        chunk.fillna(method='ffill', inplace=True)
        #chunk = chunk.iloc[33800:34200]
        #chunk=chunk.iloc[13300:15000]
        #chunk=chunk.iloc[30000:31500]
        #chunk=chunk.iloc[4800:6000]

        sum_samples += len(chunk)
        E_pred = sum(chunk.iloc[:,0])
        E_ground = sum(chunk.iloc[:,1])
        # fig, ax = plt.subplots()
        # ax.plot(chunk.index,chunk.iloc[:,0],label='pred')
        # ax.plot(chunk.index,chunk.iloc[:,1],label='true')
        # k,i=chunk.iloc[:,0][8000:9300],chunk[8000:9300].index
        # df1 = pd.DataFrame( chunk.iloc[:,0][8850:9600], index= chunk.index[8850:9600])
        # df1.to_csv("D:/Git code/daima/黄色.csv")
        # chunk.iloc[:,0][8800:9300].plot()
        # chunk.iloc[:,1][8800:9300].plot()
        chunk.iloc[:,0].plot()
        chunk.iloc[:,1].plot()

        # df1 = pd.DataFrame(chunk[8800:9300])
        # df1.to_csv("D:/Git code/daima/ukfri_recHOUSE2.csv")
        # plt.plot( chunk.iloc[:,0] )#，linewidth=5
        # plt.plot( chunk.iloc[:,1])
        # print(sum(chunk.iloc[:,0]))
        # print(sum(chunk.iloc[:,1]))
        # print()
        # ax.legend()
        # plt.show()
        # print(min(chunk.iloc[:,0]))
        # print(max(chunk.iloc[:,1]))
        chunk_results.append([
                            E_pred,
                            E_ground
                            ])
    if sum_samples == 0:
        return None
    else:
        [E_pred, E_ground] = np.sum(chunk_results,axis=0)
        #return abs(E_pred - E_ground) / float(E_ground)
        return (E_pred - E_ground)**2 / float(E_ground)**2

def SAE(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    chunk_results = []
    sum_samples = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        #chunk.fillna(method=0, inplace=True)
        sum_samples += len(chunk)
        E_pred = sum(chunk.iloc[:,0])
        E_ground = sum(chunk.iloc[:,1])
        # fig, ax = plt.subplots()
        # ax.plot(chunk.index,chunk.iloc[:,0],label='pred')
        # ax.plot(chunk.index,chunk.iloc[:,1],label='true')
        # chunk.iloc[:,0].plot()
        # chunk.iloc[:,1].plot()
        # print(sum(chunk.iloc[:,0]))
        # print(sum(chunk.iloc[:,1]))
        # print()
        # ax.legend()
        # plt.show()
        # print(min(chunk.iloc[:,0]))
        # print(max(chunk.iloc[:,1]))
        chunk_results.append([
                            E_pred,
                            E_ground
                            ])
    if sum_samples == 0:
        return None
    else:
        [E_pred, E_ground] = np.sum(chunk_results,axis=0)
        return abs(E_pred - E_ground) / float(E_ground)
def mean_squared_error(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    total_sum = 0.0
    sum_samples = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        sum_samples += len(chunk)

        total_sum += sum(((chunk.iloc[:,0]- chunk.iloc[:,1])**2))

    if sum_samples == 0:
        return None
    else:
        return total_sum / sum_samples


def mean_absolute_error(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    total_sum = 0.0
    sum_samples = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        sum_samples += len(chunk)#[5900:6050]
        total_sum += sum(abs((chunk.iloc[:,0]) - chunk.iloc[:,1]))
    if sum_samples == 0:
        return None
    else:
        return total_sum / sum_samples

def recall(tp,fn):
    return tp/float(tp+fn)

def precision(tp,fp):
    return tp/float(tp+fp)

def f1(prec,rec):
    return 2 * (prec*rec) / float(prec+rec)

def accuracy(tp, tn, p, n):
    return (tp + tn) / float(p + n)
