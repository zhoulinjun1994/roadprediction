#!/usr/bin/env python
# encoding: utf-8
# File Name: possion.py
# Author: Jiezhong Qiu
# Create Time: 2017/05/12 14:13
# TODO:

import trajectory_pb2
import weather_pb2
from datetime import datetime, timedelta
import json
import cPickle as pickle
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np


def mape_travel_time_csv(result, std):
    with open(result, "rb") as f, open(std, "rb") as fstd:
        a = [line.split(',')[-1] for line in f][1:]
        b = [line.split(',')[-1] for line in fstd][1:]
        a = np.array([float(x) for x in a])
        b = np.array([float(x) for x in b])
        nan_index = np.isnan(a) | np.isnan(b)
        a = a[~nan_index]
        b = b[~nan_index]
        return np.mean(np.abs(a-b) / b)

def parse_time(time_string):
    return datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S")

def encode_time(current_time):
    return current_time.strftime("%Y-%m-%d %H:%M:%S")

def encode_time_window(time_string):
    start_time = parse_time(time_string)
    end_time = start_time + timedelta(minutes=20)
    #[2016-10-18 08:00:00,2016-10-18 08:20:00)]
    return "\"[%s,%s)\"" % (encode_time(start_time), encode_time(end_time))


def load_trajectory(trajectory_paths, weather_path):
    data = []
    START_POINT = set([110, 120, 119, 122, 116, 111, 105, 115])
    for trajectory_path in trajectory_paths:
        with open(trajectory_path, "rb") as f:
            nu = 0
            for line in f:
                nu += 1
                if nu == 1:
                    continue
                content = [item[1:-1] for item in line.strip().split(',')]
                trajectory = trajectory_pb2.Trajectory()
                trajectory.intersection = content[0]
                trajectory.tollgate = int(content[1])
                trajectory.vehicle = int(content[2])
                trajectory.start_time = content[3]
                traces = content[4].split(';')
                for item in traces:
                    trace = trajectory.traces.add()
                    trace_content = item.split('#')
                    trace.link = int(trace_content[0])
                    trace.enter_time = trace_content[1]
                    trace.travel_time = float(trace_content[2])
                    if trace.link in START_POINT:
                        path = trajectory.paths.add()
                    path.traces.extend([trace])
                trajectory.travel_time = float(content[5])
                data.append(trajectory)
    weather_data = []
    with open(weather_path, "rb") as f:
        nu = 0
        for line in f:
            nu += 1
            if nu == 1:
                continue
            content = [item[1:-1] for item in line.strip().split(',')]
            weather = weather_pb2.Weather()
            weather.date = content[0]
            weather.hour = int(content[1])
            weather.pressure = float(content[2])
            weather.sea_pressure = float(content[3])
            weather.wind_direction = float(content[4])
            weather.wind_speed = float(content[5])
            weather.temperature = float(content[6])
            weather.rel_humidity = float(content[7])
            weather.precipitation = float(content[8])
            weather_data.append(weather)
    return data, weather_data

def poisson_parameter(data, win=1.0, prefix=""):
    link_enter_time = {}
    for trajectory in data:
        for trace in trajectory.traces:
            if trace.link not in link_enter_time:
                link_enter_time[trace.link] = []
            link_enter_time[trace.link].append(parse_time(trace.enter_time))
    for link in link_enter_time:
        link_enter_time[link] = sorted(link_enter_time[link])
        p, q = 0, 0
        time_seq = link_enter_time[link]
        print link, time_seq[0], time_seq[-1]
        x, y = [], []
        while p < len(time_seq):
            if (time_seq[-1]-time_seq[p]).total_seconds() < win*60*60:
                break
            while q+1 < len(time_seq) \
                and (time_seq[q+1]-time_seq[p]).total_seconds() <= win*60*60:
                    q += 1
            x.append((time_seq[p]-time_seq[0]).total_seconds()/ 60./60.)
            y.append(q-p+1)
            p += 1
            while p < len(time_seq) and time_seq[p] == time_seq[p-1]:
                p += 1
        with open("json/%d_%s.json" % (link, prefix), "wb") as f:
            json.dump((link, x, y), f)

def regression(trajectory_data, weather_data, output_file, sample_file, my_file):
    topology = {110108 : (110108,),
            120117 : (120117, 110118,), \
            119118: (119118, 110108), 122122 : (122122, 119118, 111103),
            116113: (116113, 111103), 111103 : (111103, 115112, 105100),
            115112 : (115112,), 105100: (105100,)
            }
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

    with open(output_file, "wb") as fout:
        for time_id in travel_time:
            for path_id, v in travel_time[time_id].iteritems():
                avg_travel_time = sum(v) / float(len(v))
                print >> fout, ','.join([str(time_id), str(path_id), str(avg_travel_time)])
    # let us take 119118 as example
    # 7.19 - 10.10 for training
    # the rest for test
    mapes = []
    result = {}
    for path_id, neighbor_path_id in topology.iteritems():
        if path_id == 120117:
            continue
        y_train, X_train = [], []
        y_test, X_test, time_test = [], [], []
        for time_id in travel_time:
            if path_id not in travel_time[time_id]:
                continue
            current_time = parse_time(time_id)
            if current_time >= datetime(2016, 10, 18) and \
                    ((current_time.hour >= 8 and current_time.hour < 10) or \
                    (current_time.hour >= 17 and current_time.hour < 19)):
                if current_time.hour < 10:
                    last_time = datetime(current_time.year, current_time.month,
                            current_time.day, 7, 40)
                    last_time_id = encode_time(last_time)
                else:
                    last_time = datetime(current_time.year, current_time.month,
                            current_time.day, 16, 40)
                    last_time_id = encode_time(last_time)
            else:
                last_time_id = encode_time(current_time - timedelta(minutes=20))
            if last_time_id not in travel_time:
                continue
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
            v = travel_time[time_id][path_id]
            if current_time >= datetime(2016, 10, 18) and \
                    ((current_time.hour >= 8 and current_time.hour < 10) or \
                    (current_time.hour >= 17 and current_time.hour < 19)):
                y_test.append(sum(v) / float(len(v)))
                X_test.append(x)
                time_test.append(time_id)
            else:
                y_train.append(sum(v) / float(len(v)))
                X_train.append(x)
        print "train linear model for %d" % path_id
        #model = LinearRegression()
        model = Ridge(alpha=1.)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        #  mape = []
        with open("linear_regression_predict_%d.csv" % path_id, "wb") as fout:
            for i in xrange(len(X_test)):
                print >> fout, "%s,%.2f,%.2f" % (time_test[i], y_pred[i], y_test[i])
                if time_test[i] not in result:
                    result[time_test[i]] = {}
                result[time_test[i]][path_id] = y_pred[i]
    #              mape.append(abs(y_pred[i] - y_test[i])/y_test[i])
    #      print "mape for %d" % path_id
    #      print sum(mape) / float(len(mape))
    #      mapes += mape
    #      with open("regression_%d.pkl" % path_id, "wb") as fout:
    #          pickle.dump((y_train, X_train, y_test, X_test), fout, protocol=pickle.HIGHEST_PROTOCOL)
    #  print "overall mape"
    #  print sum(mapes) / float(len(mapes))
    travel_time = {}
    with open(my_file, "wb") as fout, open(sample_file, "rb") as fin:
        for time_id, v in result.iteritems():
            if 110108 in v and 120117 in v:
                #  print >> fout, "A,2,%s,%.2f" % \
                #          (encode_time_window(time_id), v[110108]+v[120117])
                k = "A,2,%s" % encode_time_window(time_id)
                travel_time[k] = v[110108] + v[120117]
            if 110108 in v and 119118 in v and 122122 in v:
                #  print >> fout, "A,3,%s,%.2f" % \
                #          (encode_time_window(time_id), \
                #          v[110108]+v[119118]+v[122122])
                k = "A,3,%s" % encode_time_window(time_id)
                travel_time[k] = v[110108]+v[119118]+v[122122]
            if 105100 in v and 111103 in v and 116113 in v:
                #  print >> fout, "B,1,%s,%.2f" % \
                #          (encode_time_window(time_id), \
                #          v[105100]+v[111103]+v[116113])
                k = "B,1,%s" % encode_time_window(time_id)
                travel_time[k] = v[105100]+v[111103]+v[116113]
            if 105100 in v and 111103 in v and 122122 in v:
                #  print >> fout, "B,3,%s,%.2f" % \
                #          (encode_time_window(time_id), \
                #          v[105100]+v[111103]+v[122122])
                k = "B,3,%s" % encode_time_window(time_id)
                travel_time[k] = v[105100]+v[111103]+v[122122]
            if 115112 in v and 111103 in v and 116113 in v:
                #  print >> fout, "C,1,%s,%.2f" % \
                #          (encode_time_window(time_id), \
                #          v[115112]+v[111103]+v[116113])
                k = "C,1,%s" % encode_time_window(time_id)
                travel_time[k] = v[115112]+v[111103]+v[116113]
            if 115112 in v and 111103 in v and 122122 in v:
                #  print >> fout, "C,3,%s,%.2f" % \
                #          (encode_time_window(time_id), \
                #          v[115112]+v[111103]+v[122122])
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
    data, weather_data = load_trajectory(["../data/training/trajectories_training.csv", "../data/testing_phase1/trajectories_test1.csv"], \
            "../data/training/weather_training.csv")
    regression(data, weather_data, "../data/middle/path_average_travel_time.csv", "../result/knn_zhou_final.csv", "../result/linear_qiu_final.csv")
    #poisson_parameter(data, win=1.0, prefix="1h")

