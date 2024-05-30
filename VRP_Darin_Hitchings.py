# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:29:55 2024
@author: Darin
"""

import os
import sys
import io
import math
import time
import copy

import timeit

import numpy as np
import scipy

#from operator import itemgetter

from collections import deque

#from argparse import ArgumentParser

from glpk import glpk, GLPK

import itertools

import heapq
from queue import PriorityQueue

#runfile('VRP_Darin_Hitchings.py', args='"C:\\Users\\Darin\\Documents\\Python\\VortoInterview\\Training Problems\\problem1.txt"')

#python VRP_Darin_Hitchings.py "C:\Users\Darin\Documents\Python\VortoInterview\Training Problems\problem3.txt"

import re

import random

#import subprocess
#from multiprocessing import Process, Queue

import matplotlib.pyplot as plt

depot_id = 1e6
NumHistogramBins = 5

################################################################################################

def parse_coordinates(string):

    x = []; y = []

    #string = "(-9.100071078494038 , -48.89301103772511)" # test input with spaces

    result = re.search(r"[0-9.-]+\s*,\s*[0-9.-]+", string)

    if(result != None):
        arg_string = result.group()

        arg_string = arg_string.split(",")

        x = float(arg_string[0])
        y = float(arg_string[1])
    else:
        raise ValueError("Bad data: no arguments after command string")

    return x, y

################################################################################################

def load_input(filename):

    # Using readline()
    file1 = open(filename, 'r')
    count = 0

    customers = []
    distance_matrix = []


    while True:

        # Get next line from file
        line = file1.readline()

        # if line is empty
        # end of file is reached
        if not line:
            break


        if(line.split()[0] == "loadNumber"):
            continue


        count += 1
        #print("Line{}: {}".format(count, line.strip()))

        tokens = line.split()

        customer_number = int(tokens[0])
        coordinate_1_str = tokens[1]
        coordinate_2_str = tokens[2]

        [pickup_x,pickup_y] = parse_coordinates(coordinate_1_str)
        [dropoff_x,dropoff_y] = parse_coordinates(coordinate_2_str)

        customers.append({'number': customer_number, 'pickup': np.array([pickup_x,pickup_y]), 'dropoff': np.array([dropoff_x,dropoff_y]) })

        distance_matrix.append([pickup_x,pickup_y,dropoff_x,dropoff_y])

        #customers.append({count, })

    file1.close()

    distance_matrix = np.array(distance_matrix)

    return customers, distance_matrix

################################################################################################

def plot_graph(distance_matrix, b_pickup = True, b_dropoff = True, ids_to_plot = None):

    customer_count = distance_matrix.shape[0]

    x_pickup = distance_matrix[:,0]
    y_pickup = distance_matrix[:,1]

    x_dropoff = distance_matrix[:,2]
    y_dropoff = distance_matrix[:,3]

    # visualization : plotting with matplolib
    fig = plt.figure(figsize=(10,10))

    min_x = np.min( (np.min(distance_matrix[:,0]), np.min(distance_matrix[:,2]) )) - 10
    min_y = np.min( (np.min(distance_matrix[:,1]), np.min(distance_matrix[:,3]) )) - 10

    max_x = np.max( (np.max(distance_matrix[:,0]), np.max(distance_matrix[:,2]) )) + 10
    max_y = np.max( (np.max(distance_matrix[:,1]), np.max(distance_matrix[:,3]) )) + 10

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    plt.scatter(0.0, 0.0, c='green', s=60)
    plt.text(-0.25, -10, "depot", fontsize=10)

    if ids_to_plot == None:
        ids_to_plot = [ii for ii in range(customer_count)]

    for i in range(len(ids_to_plot)):

        id = ids_to_plot[i]

        if b_pickup:
            plt.scatter(x_pickup[id], y_pickup[id], c='orange', s=30)
            plt.text(x_pickup[id]-0.25, y_pickup[id]-10, str(id), fontsize=10)

        if b_dropoff:
            plt.scatter(x_dropoff[id], y_dropoff[id], c='blue', s=30)
            plt.text(x_dropoff[id]-0.25, y_dropoff[id]-10, str(id), fontsize=10)

    plt.axis('square')
    #plt.show()

    return fig

################################################################################################


def draw_grid_lines_and_bins(xedges, yedges, distance_matrix, x_bin_assignment, y_bin_assignment, display_title):

    eX = 0
    eY = 1

    num_clients = distance_matrix.shape[0]

    if display_title is not None:
        plt.title(display_title)

    for i in range(len(xedges)):

        plt.plot([xedges[0]-10,xedges[-1]+10],[yedges[i],yedges[i]], color='k')

    for i in range(len(yedges)):

        plt.plot([xedges[i],xedges[i]], [yedges[0]-10,yedges[-1]+10], color='k')


    for i in range(num_clients):

        plt.text(distance_matrix[i,eX]-8, distance_matrix[i,eY]+5, "(x=%d,y=%d)" % (x_bin_assignment[i], y_bin_assignment[i]), fontsize=9)

################################################################################################

"""

find_neighboring_points()

Outputs:

neighboring_points: a sparse 2D array with adjacency matrix information such that if neighboring_points(i,j) = 1 then location i and location j are close to each other
bin_2_id_map: a mapping that describes which point indices are contained within each histogram bin
lexicographic_id: the lexicographic bin assignment of each point where the 2D index is collapsed lexicographically into 1D using e.g. index = y_index*M + x_index
                  where M is a sufficiently large number to keep the x's and y's orthogonal to each other ie in separate numerical ranges that don't overlap
"""

def find_neighboring_points(distance_matrix,xedges,yedges, dx=1, dy=1, b_display=False, b_debug=False, display_title = None):

    eX = 0
    eY = 1

    M = 1000

    x_bin_assignment = np.digitize(distance_matrix[:,eX],xedges)
    y_bin_assignment = np.digitize(distance_matrix[:,eY],yedges)

    lexicographic_id = y_bin_assignment*M +x_bin_assignment

    if b_display:
        draw_grid_lines_and_bins(xedges, yedges, distance_matrix, x_bin_assignment, y_bin_assignment, display_title)



    bin_2_id_map = {}

    neighboring_points = scipy.sparse.lil_matrix((num_clients,num_clients))
    #neighboring_points = np.zeros((num_clients,num_clients))


    for i in range(len(x_bin_assignment)):

        id = y_bin_assignment[i]*M +x_bin_assignment[i]

        if id in bin_2_id_map.keys():
            bin_2_id_map[id].add(i)
        else:
            bin_2_id_map[id] = {i}


    id_offset = []

    for y_delta in range(0-dy,0+dy+1):
        for x_delta in range(0-dx,0+dx+1):
            id_offset.append((y_delta)*M + (x_delta))

    id_offset = np.array(id_offset)

    for i in range(len(lexicographic_id)):

        id = lexicographic_id[i]


        for o in range(len(id_offset)):

            neighbor_bin_id = id + id_offset[o]

            if neighbor_bin_id in bin_2_id_map:
                for n in bin_2_id_map[neighbor_bin_id]:

                    if(n != i):

                        if b_debug:

                            x_bin = id % M
                            y_bin = id // M

                            neighbor_x_bin = neighbor_bin_id % M
                            neighbor_y_bin = neighbor_bin_id // M

                            print("(%d): point(%0.2f,%0.2f) bin(%d,%d)  is close to  (%d): point(%0.2f,%0.2f)  bin(%d,%d) \n" %
                                  (i, distance_matrix[i,eX],distance_matrix[i,eY], x_bin, y_bin,
                                   n, distance_matrix[n,eX],distance_matrix[n,eY], neighbor_x_bin, neighbor_y_bin))

                        #neighboring_points[i][n] = 1
                        #neighboring_points[n][i] = 1
                        neighboring_points[i,n] = 1
                        neighboring_points[n,i] = 1


    return neighboring_points, bin_2_id_map, lexicographic_id

################################################################################################

def distance(src, dest):

    if src.ndim > 1 or dest.ndim > 1:
        d = np.sum((dest - src)**2,1)**0.5
    else:
        d = np.sum((dest - src)**2)**0.5

    return d

################################################################################################

def get_or_compute_dist(key, distance_map, distance_matrix):

    eX1 = 0; eY1 = 1

    if key in distance_map:
        d = distance_map[key]
    else:

        key0_col = 0
        key1_col = 0

        key0_modified = key[0]
        key1_modified = key[1]

        if(key[0] >= 1000):
            key0_modified = key0_modified-1000
            key0_col = 1 # switching from pair of pickup coordinates to pair of dropoff coordinates

        if(key[1] >= 1000):
            key1_modified = key1_modified-1000
            key1_col = 1 # switching from pair of pickup coordinates to pair of dropoff coordinates

        p1 = distance_matrix[key0_modified,(eX1+key0_col*2):(eY1+key0_col*2+1)]
        p2 = distance_matrix[key1_modified,(eX1+key1_col*2):(eY1+key1_col*2+1)]

        d = distance(p1, p2)

        distance_map[key[0], key[1]] = d
        distance_map[key[1], key[0]] = d

    return d, distance_map

################################################################################################

def path_str(path, distance_matrix):

    eX1 = 0; eY1 = 1
    eX2 = 2; eY2 = 3

    str = ""

    for p in path:
        if p == depot_id:
            label = 'depot'
            x = 0; y = 0
        else:
            if p > 0:
                label = 'pickup_%d' % p
                x = distance_matrix[p,eX1]
                y = distance_matrix[p,eY1]
            else:
                p = -p
                label = 'dropoff_%d' % p
                x = distance_matrix[p,eX2]
                y = distance_matrix[p,eY2]


        substr = '%s:(%0.2f, %0.2f) ' % (label, x, y)

        str = str + substr

    return str

################################################################################################

def print_route_details(route, distance_matrix):

    str = ""

    for k in route.keys():
        if k != 'path':
            substr = '%s(%0.2f) ' % (k, route[k])
            str = str + substr

    print(str)
    print(path_str(route['path'], distance_matrix))

################################################################################################

def route_permutations(number):

    pickup = np.array([i for i in range(0,number)])
    dropoff = pickup + 1000

    starting_order = np.hstack((pickup, dropoff))

    permutations = list(itertools.permutations(starting_order))

    validated_permutations = []

    for i in range(len(permutations)):

        pickups_completed = np.zeros(number)

        valid_permutation = True

        for j in range(len(permutations[i])):

            if(permutations[i][j] >= 1000):

                pickup_id = permutations[i][j] - 1000

                if(pickups_completed[pickup_id] != 1):
                    valid_permutation = False
                    break
            else:
                pickups_completed[permutations[i][j] ] = 1

        if(valid_permutation):
            validated_permutations.append(list(permutations[i]))


    return validated_permutations


################################################################################################

# I spent a bunch of time getting into the combinatorics of how pickups and dropoffs occured because
# if picking up and dropping off can be interleved bewteen clients (ie if the trucks have no capacity constraints)
# then it can make a big difference in overall route length.  Then after spending a bunch of time, I
# realized that the way our output is being printed to console makes it impossible to specify whether or not
# e.g. we have [ pickup_1 dropoff_1 pickup_2 dropoff_2 ] or [ pickup_1 pickup_2 dropoff_1 dropoff_2 ]
# or [ pickup_1 pickup_2 dropoff_2 dropoff_1 ] etc... There are actually valid 6 combinations with a pair of
# pickups/dropoffs, and more for larger groupings.  I was planning on looking at all combinations of 4 pickups+dropoffs
# After realizing that I was going outside of the problem scope, however, I stopped in the middle of doing this
# work.  I got fairly far into it though.  In real life, one would care... and it'd be worth the effort.

def first_approach(distance_matrix):


    eX = 0
    eY = 1

    eX1 = 0
    eY1 = 1
    eX2 = 2
    eY2 = 3


    #[h, xedges, yedges] = np.histogram2d(distance_matrix[:,eX], distance_matrix[:,eY],NumHistogramBins)
    #xedges = np.histogram_bin_edges(distance_matrix[:,[eX1,eX2]].ravel(),NumHistogramBins)
    #yedges = np.histogram_bin_edges(distance_matrix[:,[eY1,eY2]].ravel(),NumHistogramBins)
    xedges = np.histogram_bin_edges(distance_matrix[:,eX1],NumHistogramBins)
    yedges = np.histogram_bin_edges(distance_matrix[:,eY1],NumHistogramBins)


    dx = 3
    dy = 3


    display_title = "pickup"
    [pickup_pt_neighbors, pickup_pt_bin_2_id_map, pickup_pt_lexicographic_ids] = find_neighboring_points(distance_matrix[:,eX1:eY1+1],xedges,yedges, dx=dx, dy=dy, b_display=True, b_debug=True, display_title=display_title)
    plt.show();

    fig = plot_graph(distance_matrix, False, True)

    xedges = np.histogram_bin_edges(distance_matrix[:,eX2],NumHistogramBins)
    yedges = np.histogram_bin_edges(distance_matrix[:,eY2],NumHistogramBins)

    display_title = "dropoff"
    [dropoff_pt_neighbors, dropoff_pt_bin_2_id_map, dropoff_pt_lexicographic_ids] = find_neighboring_points(distance_matrix[:,eX2:eY2+1],xedges,yedges, dx=dx, dy=dy, b_display=True, b_debug=True, display_title=display_title)
    plt.show();


    DriverShiftLength = 60*12

    zero_matrix = np.array((num_clients, 2))

    depot_to_pickup_times = distance(zero_matrix, distance_matrix[:,eX1:eY1+1])
    pickup_to_dropoff_times = distance(distance_matrix[:,eX1:eY1+1], distance_matrix[:,eX2:eY2+1])
    dropoff_to_depot_times = distance(distance_matrix[:,eX2:eY2+1], zero_matrix)

    total_times = depot_to_pickup_times + pickup_to_dropoff_times + dropoff_to_depot_times

    worst_case_cumulative_time = np.sum(total_times)

    distance_map = {}


    for i in range(0,num_clients):

        distance_map[(depot_id, i)] = depot_to_pickup_times[i]
        distance_map[(i, depot_id)] = distance_map[(depot_id, i)]

        distance_map[(i, i+1000)] = pickup_to_dropoff_times[i]
        distance_map[(i+1000, i)] = distance_map[(i, i+1000)]

        distance_map[(i+1000, depot_id)] = dropoff_to_depot_times[i]
        distance_map[(depot_id, i+1000)] = distance_map[(i+1000, depot_id)]



    unassigned_jobs = deque()

    for i in range(0,num_clients):
        unassigned_jobs.append(i)


    remaining_time = 0

    route_list = []

    perms_of_2 = route_permutations(2) # 6
    perms_of_3 = route_permutations(3) # 90
    #perms_of_4 = route_permutations(4) # 2520, ~2 sec
    #perms_of_5 = route_permutations(5) # 113400, ~2 min

    fig = plot_graph(distance_matrix, True, True)

    # xedges = np.histogram_bin_edges(distance_matrix[:,eX1],NumHistogramBins)
    # yedges = np.histogram_bin_edges(distance_matrix[:,eY1],NumHistogramBins)
    # x_bin_assignment = np.digitize(distance_matrix[:,eX1],xedges)
    # y_bin_assignment = np.digitize(distance_matrix[:,eY1],yedges)

    # draw_grid_lines_and_bins(xedges, yedges, distance_matrix, x_bin_assignment, y_bin_assignment, "Pickup + Dropoff")

    # xedges = np.histogram_bin_edges(distance_matrix[:,eX2],NumHistogramBins)
    # yedges = np.histogram_bin_edges(distance_matrix[:,eY2],NumHistogramBins)
    # x_bin_assignment = np.digitize(distance_matrix[:,eX2],xedges)
    # y_bin_assignment = np.digitize(distance_matrix[:,eY2],yedges)

    # draw_grid_lines_and_bins(xedges, yedges, distance_matrix, x_bin_assignment, y_bin_assignment, None)

    plt.show()


    while remaining_time < DriverShiftLength:

        active_job = unassigned_jobs.popleft()

        path = [depot_id,active_job,-active_job,depot_id]
        depot_to_pickup = depot_to_pickup_times[active_job]
        pickup_to_dropoff = pickup_to_dropoff_times[active_job]
        dropoff_to_depot = dropoff_to_depot_times[active_job]

        total_time = total_times[active_job]

        I,J = pickup_pt_neighbors[active_job].nonzero()
        neighbors1 = list(J)

        I,J = dropoff_pt_neighbors[active_job].nonzero()
        neighbors2 = list(J)

        neighbors = neighbors1 + neighbors2

        permutation_ranking = []

        #for n in pickup_pt_neighbors[active_job].nonzero():
        for n in neighbors:
            #investigating detours

            #candidates:  depot -> pickup1 -> dropoff1 -> pickup2 -> dropoff2 -> depot
            #candidates:  depot -> pickup1 -> pickup2 -> dropoff1 -> dropoff2 -> depot
            #candidates:  depot -> pickup1 -> pickup2 -> dropoff2 -> dropoff1 -> depot

            #candidates:  depot -> pickup2 -> pickup1 -> dropoff1 -> dropoff2 -> depot
            #candidates:  depot -> pickup2 -> pickup1 -> dropoff2 -> dropoff1 -> depot
            #candidates:  depot -> pickup2 -> dropoff2 -> pickup1 -> dropoff1 -> depot

            id_mapping = { 0: active_job, 0+1000: active_job + 1000, 1: n, 1+1000: n+ 1000}
            #id_mapping = [ active_job, n ]

            for p in range(len(perms_of_2)):

                perm = perms_of_2[p]

                key = (depot_id, id_mapping[perm[0]])
                [d, distance_map] = get_or_compute_dist(key, distance_map, distance_matrix)

                perm_dist = d

                for k in range(len(perm)-1):
                    arc = perm[k:k+2]
                    key = (id_mapping[arc[0]], id_mapping[arc[1]])

                    [d, distance_map] = get_or_compute_dist(key, distance_map, distance_matrix)
                    perm_dist = perm_dist + d


                assert id_mapping[perm[-1]] >= 1000, "must end at a dropoff location or something is broken"

                key = (id_mapping[perm[-1]], depot_id)
                [d, distance_map] = get_or_compute_dist(key, distance_map, distance_matrix)
                perm_dist = perm_dist + d

                heapq.heappush(permutation_ranking, (perm_dist, p, 2))

            print("permutation_ranking:")
            print(permutation_ranking)
            print("Best permutation:")
            print_best_permutation()

            #permutation_ranking[0], id_mapping, perms_of_2

        route = { 'path' : path, 'depot_to_pickup' : depot_to_pickup, 'pickup_to_dropoff' : pickup_to_dropoff, 'dropoff_to_depot' : dropoff_to_depot, 'total_time' : total_time }

        route_list.append(route)

        print_route_details(route, distance_matrix)


################################################################################################

def print_queue(q1):

    #https://stackoverflow.com/questions/32488533/how-to-clone-a-queue-in-python
    q2 = PriorityQueue()
    q2.queue = copy.deepcopy(q1.queue)

    while (q2.qsize() > 0):
        top = q2.get()
        print("(%0.2f, %s, %d)" % (top[0], top[1], top[2]))

################################################################################################

def get_neighbors(pt_neighbors, trajectory):
    #[I,J]=pt_neighbors[trajectory[-1]].nonzero()
    [J]=pt_neighbors[trajectory[-1]].nonzero()
    neighbors = J

    return neighbors

################################################################################################

def run_glpk_solver(num_clients, num_strategies, all_ids, keys, validPaths_sorted, b_display, b_debug):

    ConstraintMatrix = np.zeros((num_clients, num_strategies))
    objective_fx = np.zeros(num_strategies)


    for i in range(num_strategies):

        if(len(keys[i]) > 1):
            for j in range(len(keys[i])):

                ConstraintMatrix[keys[i][j],i] = 1
        else:
            ConstraintMatrix[keys[i],i] = 1

        objective_fx[i] = validPaths_sorted[keys[i]][0]


    b_eq = [1 for i in range(num_clients)]


    bincon = [ i for i in range(num_strategies) ] # indices of binary decision variables for GLPK
    mip_options = { 'bincon' : bincon, 'proxy' : 30, 'tol_int' : 1e-7, 'tol_obj' : 1e-9, 'cuts' : 'all', 'round' : False }

    basis_fac = "luf+ft"
    scale = False # turning scaling off to suppress the associated glpk output to stdout
    maxit = 1e7
    timeout = 30

    solver_options = 'mip'

    Bounds_GLPK = []

    #for i in range(num_strategies):
    #    Bounds_GLPK.append( (0, 1) ) # I think the variable bounds are automatically implied because all variables here are binary decision variables

    if b_debug == True:
        message_level = 3 # monotonically increasing verbosity
        glpk_disp = True
    else:
        message_level = 0
        glpk_disp = False


    glpk_res = glpk(c = objective_fx, A_ub = None, b_ub = None, A_eq = ConstraintMatrix, b_eq = b_eq, bounds = Bounds_GLPK, solver = solver_options, sense = GLPK.GLP_MIN, scale = scale,
         maxit = maxit, timeout = timeout, basis_fac = basis_fac, message_level = message_level, disp = glpk_disp, simplex_options=None,
         ip_options = None, mip_options = mip_options)



    assert(glpk_res.success == True), "Failed to find solution"

    strategies_employed = np.where(glpk_res.x)[0]

    final_strategies = []

    ids_employed_check = set()

    for i in range(len(strategies_employed)):

        assert objective_fx[strategies_employed[i]] == validPaths_sorted[keys[strategies_employed[i]]][0], "objective_fx[strategies_employed[i]] != validPaths_sorted[keys[strategies_employed[i]]][0]"

        key_index = strategies_employed[i]

        if b_debug:
            print(validPaths_sorted[keys[key_index]])

        path = validPaths_sorted[keys[strategies_employed[i]]]

        for id in keys[strategies_employed[i]]:
            ids_employed_check.add( id )

        final_strategies.append(path)

    forgotten_ids = list(set(all_ids) - ids_employed_check)
    assert len(forgotten_ids) == 0, "len(forgotten_ids) != 0"

    return final_strategies, glpk_res

################################################################################################

if __name__ == '__main__':
    #parser = ArgumentParser()
    #parser.add_argument("-p", "--path", dest="filename",
    #                    help="input FILE", metavar="FILE")

    #args = parser.parse_args()

    b_debug = False
    b_display = False

    max_neighbors = 50
    QueueSizeStoppingPoint = 50e3
    MaxDepth = 6

    random.seed(0)

    if(len(sys.argv) > 1):
        argv = sys.argv[1]
    else:
        #filenumber = 1 # 10 destinations
        #filenumber = 2 # 50 destinations
        filenumber = 12 # 200 destinations
        argv="C:\\Users\\Darin\\Documents\\Python\\VortoInterview\\Training Problems\\problem%d.txt" % filenumber

    filename = argv

    customers, distance_matrix = load_input(filename)

    if b_display:
        fig = plot_graph(distance_matrix, True, True, [7,12,19,22,25,35,86,39,94,44,81])
        plt.show()

    eX = 0
    eY = 1

    eX1 = 0
    eY1 = 1
    eX2 = 2
    eY2 = 3

    if b_debug:

        for c in customers:
            print("%d (%0.4f, %0.4f) (%0.4f, %0.4f) \n" % (c['number'], c['pickup'][eX], c['pickup'][eY], c['dropoff'][eX], c['dropoff'][eY]))


    DriverShiftLength = 60*12
    num_clients = distance_matrix.shape[0]

    zero_matrix = np.zeros((num_clients, 2))

    depot_to_pickup_times = distance(zero_matrix, distance_matrix[:,eX1:eY1+1])
    #depot_to_pickup_times2 = np.linalg.norm(distance_matrix[:,eX1:eY1+1] - zero_matrix,axis=1)
    #assert(np.all(depot_to_pickup_times == depot_to_pickup_times2)), "math is broken"

    pickup_to_dropoff_times = distance(distance_matrix[:,eX1:eY1+1], distance_matrix[:,eX2:eY2+1])
    #pickup_to_dropoff_times2 = np.linalg.norm(distance_matrix[:,eX2:eY2+1] - distance_matrix[:,eX1:eY1+1],axis=1)
    #assert(np.all(pickup_to_dropoff_times == pickup_to_dropoff_times2)), "math is broken"

    dropoff_to_depot_times = distance(distance_matrix[:,eX2:eY2+1], zero_matrix)
    #dropoff_to_depot_times2 = np.linalg.norm(zero_matrix - distance_matrix[:,eX2:eY2+1],axis=1)
    #assert(np.all(dropoff_to_depot_times == dropoff_to_depot_times2)), "math is broken"

    total_times = depot_to_pickup_times + pickup_to_dropoff_times + dropoff_to_depot_times



    #################################
    xedges_pickup = np.histogram_bin_edges(distance_matrix[:,eX1],NumHistogramBins)
    yedges_pickup = np.histogram_bin_edges(distance_matrix[:,eY1],NumHistogramBins)

    if NumHistogramBins == 5:

        dx = 1
        dy = 1  # this option covers all points in an area of 3x3 cells out of 5x5

    elif NumHistogramBins == 7:
        dx = 2
        dy = 2  # this option covers all points in an area of 5x5 cells out of 7x7
    else:
        assert NumHistogramBins == 9, "expecting NumHistogramBins == 9 or NumHistogramBins == 7 or NumHistogramBins == 5"
        dx = 3 # this option covers all points in an area of 7x7 cells out of 9x9
        dy = 3

    b_timing = False

    if b_timing:
        t_0 = timeit.default_timer()
        b_display=False
        b_debug=False

    display_title = "pickup"
    [pickup_pt_neighbors, pickup_pt_bin_2_id_map, pickup_pt_lexicographic_ids] = find_neighboring_points(distance_matrix[:,eX1:eY1+1],xedges_pickup,yedges_pickup, dx=dx, dy=dy, b_display=False, b_debug=False, display_title=display_title)

    xedges_dropoff = np.histogram_bin_edges(distance_matrix[:,eX1],NumHistogramBins)
    yedges_dropoff = np.histogram_bin_edges(distance_matrix[:,eY1],NumHistogramBins)

    display_title = "dropoff"
    [dropoff_pt_neighbors, dropoff_pt_bin_2_id_map, dropoff_pt_lexicographic_ids] = find_neighboring_points(distance_matrix[:,eX2:eY2+1],xedges_dropoff,yedges_dropoff, dx=dx, dy=dy, b_display=False, b_debug=False, display_title=display_title)

    pt_neighbors_temp = pickup_pt_neighbors + dropoff_pt_neighbors # WARN: slow for lil_matrix

    neighbor_counts = np.sum(pt_neighbors_temp > 0,1)


    #pt_neighbors = scipy.sparse.lil_matrix((num_clients,num_clients))
    pt_neighbors = np.zeros((num_clients,num_clients))

    for i in range(num_clients):
        I,J = pt_neighbors_temp[i].nonzero()

        if np.sum(J) > max_neighbors:

            threshold = max_neighbors / neighbor_counts[i,0]

            for j in range(len(J)):

                if j < i:
                    continue

                if random.uniform(0,1) < threshold:

                    pt_neighbors[i][J[j]] = 1
                    pt_neighbors[J[j]][i] = 1



    if b_timing:
        t_1 = timeit.default_timer()
        elapsed_time = round((t_1 - t_0) * 10 ** 3, 3)
        print(f"Elapsed time: {elapsed_time} ms")

    #if b_display:
    #    plt.show(); # needed when there's plotting activity in find_neighboring_points()

    #################################



    i = 0
    all_ids = [i for i in range(num_clients)]

    closedList = {}
    validPaths = {}
    #openList = []
    #openList_next = []

    openList = PriorityQueue()
    openList_next = PriorityQueue()


    for i in range(num_clients):
        #heapq.heappush(openList, Path(total_times[i], [i], 1))
        neighbors = get_neighbors(pt_neighbors, [i])

        #neighbor_count = np.min((len(neighbors), max_neighbors))
        neighbor_count = len(neighbors)

        openList.put((total_times[i], [i], 1, set(neighbors[0:neighbor_count])))
        validPaths[tuple([i])] = total_times[i] # mapping from a path (as a sequence of clients to visit) to the time the path takes

    done = False
    count = 0

    eTime = 0
    eTraj = 1
    eDepth = 2
    eNeighbors = 3

    while done == False:
        while openList.qsize() > 0:

            #top = heapq.heappop(openList)
            top = openList.get()

            #trajectory = top.traj
            trajectory = top[eTraj]

            #if(trajectory == [48 ]):
            #    print("hit test condition")

            #neighbors = list(set(all_ids) - set(trajectory))

            neighbors_set = top[eNeighbors] - set(trajectory)
            neighbors = list(neighbors_set)

            if b_debug:
                print("trajectory: %s   neighbors: %s" % (trajectory,neighbors));

            terminations = 0

            for n in range(len(neighbors)):

                #if(trajectory == [48 ] and neighbors[n] == 178):
                #    print("hit test condition")

                #elapsed_time = top.time
                elapsed_time = top[eTime]

                dropoff1 = distance_matrix[trajectory[-1],eX2:eY2+1]
                pickup2 = distance_matrix[neighbors[n],eX1:eY1+1]

                pickup2_time_increment = distance(dropoff1, pickup2) # the first of the 3 legs that are added, I back out the other too from the total time of the 2nd path
                pickup2_orig_time = depot_to_pickup_times[neighbors[n]]

                new_time = elapsed_time - dropoff_to_depot_times[trajectory[-1]] + pickup2_time_increment + total_times[neighbors[n]] - pickup2_orig_time

                #if(len(trajectory) == 3):
                #    print("hit test condition")

                if(new_time  < 12*60 ):

                    if(len(trajectory) < MaxDepth):
                        new_traj = trajectory + [neighbors[n]]

                        #if(new_traj == [48, 178]):
                        #    print("hit test condition")

                        #if(new_traj == [48, 178, 132]):
                        #    print("hit test condition")

                        #heapq.heappush(openList_next, (new_time, new_traj, top.depth+1))
                        openList_next.put( (new_time, new_traj, top[eDepth]+1, neighbors_set - {n}) )

                        validPaths[tuple(new_traj)] = new_time
                    else:
                        terminations = terminations + 1
                else:
                    terminations = terminations + 1

            #if terminations == len(neighbors):
            #    print("ending trajectory %s" % trajectory)
            #    closedList[tuple(trajectory)] = elapsed_time


        openList = openList_next
        openList_next = PriorityQueue()

        count = count + 1

        if b_debug:
            print("iteration: %d   queue size: %d " % (count, openList.qsize()))

            print_queue(openList)

        if b_debug or False:
            print("qsize: %d" % openList.qsize())

        if openList.qsize() == 0 or openList.qsize() > QueueSizeStoppingPoint:
            done = True


    keys = list(validPaths.keys())

    validPaths_sorted = {}

    #https://stackoverflow.com/questions/3121979/how-to-sort-a-list-tuple-of-lists-tuples-by-the-element-at-a-given-index


    #finding whichever path has the shortest length among the sets of paths that hit the same locations but in different orders
    for i in range(len(keys)):

        key = keys[i]
        sorted_key = tuple(np.sort(key))

        #key.sort(key=itemgetter(0))

        if sorted_key in validPaths_sorted:

            if(validPaths[key] < validPaths_sorted[sorted_key][0]):
                validPaths_sorted[sorted_key] = ( validPaths[key], key )
        else:
            validPaths_sorted[sorted_key] = ( validPaths[key], key )


    keys = list(validPaths_sorted.keys())

    num_strategies = len(keys)

    gittins_index = np.zeros(num_strategies)

    for i in range(num_strategies):
        #the figure of merit being used here is the number of locations serviced divided by the total route length
        gittins_index[i] = len(keys[i]) / validPaths_sorted[keys[i]][0] # dict value contains (distance, original_key) tuple

    sorted_indices = np.flip(np.argsort(gittins_index))

    visited_clients = set()
    final_strategies = []

    for i in range(num_strategies):
        index = sorted_indices[i] # index is an index into keys[] and the strategies I've enumerated

        index_set = set(keys[index])

        if index_set & visited_clients != set():
            pass
        else:

            path = validPaths_sorted[keys[index]]
            final_strategies.append(path)

            visited_clients = visited_clients | index_set # union of 2 sets

    assert set(all_ids) - visited_clients == set(), "The set difference set(all_ids) - visited_clients is not empty, some clients were not visited"

    #The following GPLK solver method often provides the optimal solution (relative to the strategies we took the time to enumerate), however
    #I can't guarantee that it will generate no screen output without compiling the underlieing library from source and also
    #making code changes to it to suppress the verbosity level, it's not following its own conventions concerning when it prints to screen
    #and I haven't been able to run the library within a subprocess / a separate process in order to direct the stdout elsewhere.
    #Also, while for smaller problems it's giving the optimal answer, it's getting stuck on a number of the larger ones.  Ie
    #it's fine with 100s of binary decision variables, but not thousands.  There are better ways of solving this problem but
    #it will require a commercial grade MILP solver.  Also, given more time I could generate a column generation approach and/or
    #feed the solver with an initial feasible solution from which it could improve on. But evidently that capability does not
    #exist within this python wrapper library I have chosen, and I don't have time to try to add it in.

    #final_strategies, glpk_res = run_glpk_solver(num_clients, num_strategies, all_ids, keys, validPaths_sorted, b_display, b_debug)




    for i in range(len(final_strategies)):

        adjusted_indexing = np.array(final_strategies[i][1]) + 1

        #every option I use here prints a '\r' to the output, it's unavoidable, which means the evaluateShared.py has a windows compatibility bug for not stripping out '\r' as was all '\n'
        print(list(adjusted_indexing)) # ordering is [ distance (trajectory_tuple) ]
        #print(list(adjusted_indexing), end='\n')
        #strategy_string = "%s" % list(adjusted_indexing)
        #sys.stdout.write(strategy_string + '\n')


    #print("")