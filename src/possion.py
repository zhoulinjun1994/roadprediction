#!/usr/bin/env python
# encoding: utf-8
# File Name: possion.py
# Author: Jiezhong Qiu
# Create Time: 2017/05/12 14:13
# TODO:

import trajectory_pb2;
from datetime import datetime
import json

def parse_time(time_string):
    return datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S")

def load_trajectory(path):
    data = []
    with open(path) as f:
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
            trajectory.travel_time = float(content[5])
            data.append(trajectory)
    return data

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
if __name__ == "__main__":
    data = load_trajectory("../data/training/trajectories_training.csv")
    poisson_parameter(data, win=1.0, prefix="1h")
