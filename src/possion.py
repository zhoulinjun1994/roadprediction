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


def parse_time(time_string):
    return datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S")

def encode_time(current_time):
    return current_time.strftime("%Y-%m-%d %H:%M:%S")

def load_trajectory(trajectory_path, weather_path):
    data = []
    START_POINT = set([110, 120, 119, 122, 116, 111, 105, 115])
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

def regression(trajectory_data, weather_data, output_file):
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
    with open(output_file, "wb") as fout:
        for time_id in travel_time:
            for path_id, v in travel_time[time_id].iteritems():
                avg_travel_time = sum(v) / float(len(v))
                print >> fout, ','.join([str(time_id), str(path_id), str(avg_travel_time)])
    # let us take 119118 as example
    path_id = 116113
    neighbor_path_id = [116113, 119118, 111103, 122122]
    y, X = [], []
    for time_id in travel_time:
        if path_id not in travel_time[time_id]:
            continue
        current_time = parse_time(time_id)
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
        hour = [0.] * 24
        hour[current_time.hour] = 1.
        x += hour[:-1]
        v = travel_time[time_id][path_id]
        y.append(sum(v) / float(len(v)))
        X.append(x)
    with open("regression_116113.pkl", "wb") as fout:
        pickle.dump((y, X), fout, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    data, weather_data = load_trajectory("../data/training/trajectories_training.csv", \
            "../data/training/weather_training.csv")
    regression(data, weather_data, "../data/middle/path_average_travel_time.csv")
    #poisson_parameter(data, win=1.0, prefix="1h")

