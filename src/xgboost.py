import trajectory_pb2
import weather_pb2
from datetime import datetime, timedelta
import json
import cPickle as pickle
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
from possion import *

def generalize(topology, trajectory_data, weather_data):
    travel_time = {}
    for trajectory in trajectory_data:
        for path in trajectory.paths:
            enter_time = parse_time(path.traces[0].enter_time)
            current_time = datetime(enter_time.year, enter_time.month, \
                    enter_time.day, enter_time.hour, enter_time.minute/20*20)
            time_id = encode_time(current_time)
            path_id = path.traces[0].link * 1000 + path.traces[-1].link
            if time_id not in travel_time:
                travel_time[time_id] = {}
            if path_id not in travel_time[time_id]:
                travel_time[time_id][path_id] = []
            travel_time[time_id][path_id].append(sum(\
                    [trace.travel_time for trace in path.traces]))
    # generate test data manually
    for day in xrange(18, 25):
        for hour in [8, 9, 17, 18]:
            for minute in [0, 20, 40]:
                current_time = datetime(2016, 10, day, hour, minute)
                time_id = encode_time(current_time)
                if time_id not in travel_time:
                    travel_time[time_id] = {}
                for path_id in topology:
                    travel_time[time_id][path_id] = [0.0]
    
    avg_travel_info = {}
    for time in travel_time:
        avg_travel_info[time] = {}
        for path in travel_time[time]:
            cnt = float(len(travel_time[time][path]))
            avg_travel_info[time][path] = [sum(travel_time[time][path]) / cnt, cnt]
            
    return travel_time, avg_travel_info
    
def empty_road_model(avg_travel_info, path_id, start_time, end_time, min_thres, max_thres):
    y_train, X_train = [], []
    for time_id in avg_travel_info: 
        if path_id not in avg_travel_info[time_id]:
            continue
        current_time = parse_time(time_id)
        if current_time >= parse_time(start_time) and current_time < parse_time(end_time):
            if avg_travel_info[time_id][path_id][1] >= min_thres and avg_travel_info[time_id][path_id][1] <= max_thres:
                x = []
                #month = [0.] * 4
                #month[current_time.month-7] = 1.
                #x += month[:-1]
                #weekday = [0.] * 7
                #weekday[current_time.weekday()] = 1.
                #x += weekday[:-1]
                hour = [0.] * 12
                hour[current_time.hour/2] = 1.
                x += hour[:-1]
                
                y_train.append(avg_travel_info[time_id][path_id][0])
                X_train.append(x)
    y_train = np.array(y_train)
    print len(y_train)
    model = LinearRegression()
    #model = Ridge(alpha=1.)
    model.fit(X_train, y_train)
    y_est = model.predict(X_train)
    print np.sum(np.abs((y_est - y_train)) / y_train) / len(y_train)
    print y_est[:20]
    print y_train[:20]
    return model
    
def feature_construct(neighbor_path_id, travel_time, last_time_id, current_time):
    flag = True
    x = []
    for neighbor in neighbor_path_id:
        if neighbor not in travel_time[last_time_id]:
            flag = False
            break
        else:
            v = travel_time[last_time_id][neighbor]
            x.append(sum(v) / float(len(v)))
        if not flag:
            continue
    month = [0.] * 4
    month[current_time.month-7] = 1.
    x += month[:-1]
    weekday = [0.] * 7
    weekday[current_time.weekday()] = 1.
    x += weekday[:-1]
    hour = [0.] * 12
    hour[current_time.hour/2] = 1.
    x += hour[:-1]
    return flag, x
            

def regression(path_id, topology, travel_time, train_st, train_end):
    mapes = []
    result = {}
    neighbor_path_id = topology[path_id]
    y_train, X_train = [], []
    for time_id in travel_time:
        if path_id not in travel_time[time_id]:
            continue
        current_time = parse_time(time_id)
        if current_time >= parse_time(train_st) and current_time < parse_time(train_end):
            last_time_id = encode_time(current_time - timedelta(minutes=20))
            if last_time_id not in travel_time:
                continue
            
            v = travel_time[time_id][path_id]
            flag, x = feature_construct(neighbor_path_id, travel_time, last_time_id, current_time)
            if flag:
                y_train.append(sum(v) / float(len(v)))
                X_train.append(x)
    print "train linear model for %d" % path_id
    #model = LinearRegression()
    model = Ridge(alpha=1.)
    model.fit(X_train, y_train)
    return model
    
def regression_test(model, path_id, topology, travel_time, test_st, test_end, encoded_time):
    X_test, time_test = [], []
    neighbor_path_id = topology[path_id]
    for time_id in travel_time:
        if path_id not in travel_time[time_id]:
            continue
        current_time = parse_time(time_id)
        if current_time >= parse_time(test_st) and current_time < parse_time(test_end):
            last_time_id = encoded_time
            if last_time_id not in travel_time:
                continue
            
            flag, x = feature_construct(neighbor_path_id, travel_time, last_time_id, current_time)
            if flag:
                X_test.append(x)
                time_test.append(time_id)
    if len(X_test) == 0:
        return (path_id, {})
    y_pred = model.predict(X_test)
    result = {}
    for i in xrange(len(X_test)):
        if time_test[i] not in result:
            result[time_test[i]] = {}
        result[time_test[i]][path_id] = y_pred[i]
    return (path_id, result)
    
def concatenate_result(result):
    output = {}
    for i in range(len(result)):
        for time_id in result[i][1]:
            if time_id not in output:
                output[time_id] = {}
            output[time_id][result[i][0]] = result[i][1][time_id][result[i][0]]
    return output   

def print_out(result, sample_file, my_file):
    travel_time = {}
    with open(my_file, "wb") as fout, open(sample_file, "rb") as fin:
        for time_id, v in result.iteritems():
            if 110108 in v and 120117 in v:
                k = "A,2,%s" % encode_time_window(time_id)
                travel_time[k] = v[110108] + v[120117]
            if 110108 in v and 119118 in v and 122122 in v:
                k = "A,3,%s" % encode_time_window(time_id)
                travel_time[k] = v[110108]+v[119118]+v[122122]
            if 105100 in v and 111103 in v and 116113 in v:
                k = "B,1,%s" % encode_time_window(time_id)
                travel_time[k] = v[105100]+v[111103]+v[116113]
            if 105100 in v and 111103 in v and 122122 in v:
                k = "B,3,%s" % encode_time_window(time_id)
                travel_time[k] = v[105100]+v[111103]+v[122122]
            if 115112 in v and 111103 in v and 116113 in v:
                k = "C,1,%s" % encode_time_window(time_id)
                travel_time[k] = v[115112]+v[111103]+v[116113]
            if 115112 in v and 111103 in v and 122122 in v:
                k = "C,3,%s" % encode_time_window(time_id)
                travel_time[k] = v[115112]+v[111103]+v[122122]
        nu = 0
        match = 0
        for line in fin:
            nu += 1
            if nu == 1:
                print >> fout, line.strip()
                continue
            content = line.strip().split(',')
            k = ','.join(content[:4])
            if k in travel_time:
                content[4] = str(travel_time[k])
                match += 1
            print >> fout, ','.join(content)
        print "%d matches" % match
    print mape_travel_time_csv(my_file, sample_file)

if __name__ == "__main__":
    topology = {110108 : (110108,),
                120117 : (120117, 110118,), \
                119118: (119118, 110108), 122122 : (122122, 119118, 111103),
                116113: (116113, 111103), 111103 : (111103, 115112, 105100),
                115112 : (115112,), 105100: (105100,)
                }
    data, weather_data = load_trajectory(["../data/training/trajectories_training.csv", "../data/testing_phase1/trajectories_test1.csv"], \
            "../data/training/weather_training.csv")
    travel_time, avg_travel_info = generalize(topology, data, weather_data)
    #empty_road_model(avg_travel_info, 110108, '2016-07-19 00:00:00', '2016-10-17 23:59:59', 5, 10)
    #regression(topology, travel_time, "../result/knn_zhou_final.csv", "linear_qiu_final.csv")
    result = []
    for path_id in topology:
        if path_id == 120117:
            continue
        model = regression(path_id, topology, travel_time, '2016-07-19 00:00:00', '2016-10-17 23:59:59')
        for day in xrange(18, 25):
            for hour in [8, 17]:
                result.append(regression_test(model, path_id, topology, travel_time, encode_time(datetime(2016, 10, day, hour, 0)), encode_time(datetime(2016, 10, day, hour+2, 0)), encode_time(datetime(2016, 10, day, hour-1, 40))))
    output = concatenate_result(result)
    print_out(output, "../result/knn_zhou_final.csv", "linear_qiu_final.csv")