from possion import parse_time, load_trajectory
import datetime
import numpy as np

ROUTE = 6

# Route Index
def route_index(intersection, tollgate):
    if intersection == 'A' and tollgate == 2:
        return 0
    elif intersection == 'A' and tollgate == 3:
        return 1
    elif intersection == 'B' and tollgate == 1:
        return 2
    elif intersection == 'B' and tollgate == 3:
        return 3
    elif intersection == 'C' and tollgate == 1:
        return 4
    elif intersection == 'C' and tollgate == 3:
        return 5
        
# Calculating for aggregate average travel time for each time slice
# Input: time slice(in seconds), trajectory data, START_TIME(e.g. 2000-01-01 00:00:00), END_TIME
# Output: A ROUTENUM * TIMESLOTNUM matrix(list) representing for averager travel time
def calc_aggr_travel_time(slice, data, START_TIME, END_TIME):
    time_map = []    
    for slot in range(int((parse_time(END_TIME) - parse_time(START_TIME)).total_seconds() / slice)):
        time_map.append(parse_time(START_TIME) + datetime.timedelta(seconds=slot * slice))
    travel_time = [[[] for j in range(len(time_map))] for i in range(ROUTE)]
    p = 0
    for traj in data:
        s_time = parse_time(traj.start_time)
        while p < len(time_map) - 1:
            if s_time < time_map[p + 1]:
                travel_time[route_index(traj.intersection, traj.tollgate)][p].append(traj.travel_time)
                break
            else:
                p += 1
        if p == len(time_map) - 1:
            travel_time[route_index(traj.intersection, traj.tollgate)][p].append(traj.travel_time)
    avg_aggr_travel_time = [[np.average(np.array(travel_time[i][j])) for j in range(len(time_map))] for i in range(ROUTE)]
    return avg_aggr_travel_time
                

if __name__ == '__main__':
    TR_START_TIME = '2016-07-19 00:00:00'
    TR_END_TIME = '2016-10-18 00:00:00'
    TE_START_TIME = '2016-10-18 00:00:00'
    TE_END_TIME = '2016-10-25 00:00:00'
    trdata = load_trajectory("../data/training/trajectories_training.csv")
    tedata = load_trajectory("../data/testing_phase1/trajectories_test1.csv")
    avg_tr_time = calc_aggr_travel_time(20*60, trdata, TR_START_TIME, TR_END_TIME)
    avg_te_time = calc_aggr_travel_time(20*60, tedata, TE_START_TIME, TE_END_TIME)
    np.save('../data/middle/training_avg_aggr_travel_time_20min.npy', np.array(avg_tr_time))
    np.save('../data/middle/testing_avg_aggr_travel_time_20min.npy', np.array(avg_te_time))