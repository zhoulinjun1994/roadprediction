import numpy as np
import math
from possion import parse_time, load_trajectory
import time
import datetime

# Naive-KNN
def naive_knn(tr_time, te_time, route, slot, tr_start_time, tr_last_time, te_start_time, te_last_time):
    TRAINING_DAY = len(tr_time[route]) / (86400 / slot)
    TESTING_DAY = len(te_time[route]) / (86400 / slot)
    tr_start_point = tr_start_time / slot
    tr_end_point = (tr_start_time + tr_last_time) / slot - 1
    te_start_point = te_start_time / slot
    te_end_point = (te_start_time + te_last_time) / slot - 1
    S1_mat = [[] for i in range(TRAINING_DAY)]
    S2_mat = [[] for i in range(TRAINING_DAY)]
    T1_mat = [[] for i in range(TESTING_DAY)]
    T2_mat = [[] for i in range(TESTING_DAY)]
    for i in range(TRAINING_DAY):
        for t in range(tr_start_point, tr_end_point+1):
            S1_mat[i].append(tr_time[route][i * 86400 / slot + t])
        for t in range(te_start_point, te_end_point+1):
            S2_mat[i].append(tr_time[route][i * 86400 / slot + t])
    for i in range(TESTING_DAY):
        for t in range(tr_start_point, tr_end_point+1):
            T1_mat[i].append(te_time[route][i * 86400 / slot + t])
    S1 = np.array(S1_mat)
    S2 = np.array(S2_mat)
    T1 = np.array(T1_mat)
    weight = np.zeros((TESTING_DAY, TRAINING_DAY))
    for i in range(TESTING_DAY):
        for j in range(TRAINING_DAY):
            cnt = 0
            for k in range(tr_last_time / slot):
                if not math.isnan(S1[j][k]) and not math.isnan(T1[i][k]):
                    weight[i][j] += (abs(S1[j][k] - T1[i][k]) / T1[i][k])
                    cnt += 1
            if cnt == 0:
                weight[i][j] = 1.0
            else:
                weight[i][j] = max(weight[i][j] / float(cnt), 0.20)
    #print S1[10]
    #print T1[2]
    #print weight[3]
    for i in range(TESTING_DAY):
        sum_weight = 0.0
        for j in range(TRAINING_DAY):
            sum_weight += math.pow(2.0, 1.0 / weight[i][j])
        for j in range(TRAINING_DAY):
            weight[i][j] = math.pow(2.0, 1.0 / weight[i][j]) / sum_weight
    #print weight[3]
    T2 = np.zeros((TESTING_DAY, te_last_time / slot))
    for i in range(TESTING_DAY):
        for j in range(te_last_time / slot):
            prob = 0.0
            for k in range(TRAINING_DAY):
                if not math.isnan(S2[k][j]):
                    prob += weight[i][k]
                    T2[i][j] += (weight[i][k] * S2[k][j])
            T2[i][j] = T2[i][j] / prob
    return T2

# Output form, result_data is a 6*42 matrix representing for 8:00-10:00, result_data2 is also 6*42 representing for 17:00-19:00    
def printout(result_data, result_data2, filename):
    prefix = ['A,2', 'A,3', 'B,1', 'B,3', 'C,1', 'C,3']
    STARTTIME = parse_time('2016-10-18 08:00:00')
    STARTTIME2 = parse_time('2016-10-18 17:00:00')
    with open(filename, 'w') as f:
        print >>f, 'intersection_id,tollgate_id,time_window,avg_travel_time'
        for r in range(6):
            for i in range(7):
                for j in range(6):
                    s_time = STARTTIME + datetime.timedelta(seconds=86400*i+1200*j)
                    e_time = s_time + datetime.timedelta(seconds=1200)
                    print >>f, prefix[r]+',\"['+ s_time.strftime("%Y-%m-%d %H:%M:%S")+','+e_time.strftime("%Y-%m-%d %H:%M:%S")+')\",'+ str(result_data[r][i*6+j])
        for r in range(6):
            for i in range(7):
                for j in range(6):
                    s_time = STARTTIME2 + datetime.timedelta(seconds=86400*i+1200*j)
                    e_time = s_time + datetime.timedelta(seconds=1200)
                    print >>f, prefix[r]+',\"['+ s_time.strftime("%Y-%m-%d %H:%M:%S")+','+e_time.strftime("%Y-%m-%d %H:%M:%S")+')\",'+ str(result_data2[r][i*6+j])

# calc for MAPE, r1/v1 representing for 8:00-10:00 prediction/true value and r2/v2 for 17:00-19:00                    
def calc_MAPE(r1, r2, v1, v2):
    r = np.concatenate((r1, r2), axis=1)
    v = np.concatenate((v1, v2), axis=1)
    cnt = 0
    MAPE = 0.0
    H, W = np.shape(v)
    for i in range(H):
        for j in range(W):
            if math.isnan(v[i][j]):
                continue
            else:
                MAPE += (abs(v[i][j] - r[i][j]) / v[i][j])
                cnt += 1
    return MAPE / float(cnt)

if __name__ == '__main__':
    tr_time = np.load('../data/middle/training_avg_aggr_travel_time_20min.npy')
    te_time = np.load('../data/middle/testing_avg_aggr_travel_time_20min.npy')
    result_data = []
    result_data2 = []
    #T2 = naive_knn(tr_time, te_time, 0, 20*60, 6*60*60, 2*60*60, 8*60*60, 2*60*60)
    
    for i in range(6):
        T2 = naive_knn(tr_time, te_time, i, 20*60, 6*60*60, 2*60*60, 8*60*60, 2*60*60)
        result_data.append(np.reshape(T2, len(T2) * len(T2[0])))
    for i in range(6):
        T2 = naive_knn(tr_time, te_time, i, 20*60, 15*60*60, 2*60*60, 17*60*60, 2*60*60)
        result_data2.append(np.reshape(T2, len(T2) * len(T2[0])))
    printout(result_data, result_data2, 'naiveknn.csv')
    '''
    val_data = np.concatenate((te_time[:, 24:30], te_time[:, 96:102], te_time[:, 168:174], te_time[:, 240:246], te_time[:, 312:318], te_time[:, 384:390], te_time[:, 456:462]), axis=1)
    val_data2 = np.concatenate((te_time[:, 51:57], te_time[:, 123:129], te_time[:, 195:201], te_time[:, 267:273], te_time[:, 339:345], te_time[:, 411:417], te_time[:, 483:489]), axis=1)
    #printout(val_data, val_data2, 'naiveknn_valval.csv')
    
    print calc_MAPE(result_data, result_data2, val_data, val_data2)
    '''
    
    
    
    