from model.run import *

import csv
import sys
import time
import multiprocessing 
import os 
from itertools import product
from pathlib import Path


''' This script runs phase 1 of the model for different numbers of the agents and iterations 

Input:
    filename, name of the file, where the results will be stored

The results will be stored in filename.csv in the directory ./data/final

'''

nrepeats = 50
seeds = np.linspace(0,nrepeats, dtype=int, num = nrepeats)
agentNums = [10, 100, 1000]

maxItsPerA = 50

# Model Parameters
div = 0.5
utility_shape = 500
util_smoothness = 30


def runModel_timePerAgent(iteration_tuple): 
    '''
    evaluates phase 1 for different iterations per agent and number of agents

    Input: 
        iteration_tuple = (number of agents, seed for the random generator)

    Returns:
        List with each row containing the model parameters and the average utility of the best design that was created '''

    print("Worker process id for {0}: {1}".format(iteration_tuple, os.getpid()))
    ai = iteration_tuple[0]
    seed = iteration_tuple[1]
    
    rows = []
    t0 = time.perf_counter()

    # initialize the model and find the max and min average utility
    Sim = Simulation(numAgents=ai, seed=seed, diversity=div, utility_parameter=util_smoothness)
    x_max = Sim.find_maximum_design(Sim.Uavg)
    x_min = Sim.Repo[0]
    smax = Sim.Uavg[x_max]
    smin = Sim.Uavg[x_min]
    t1 = time.perf_counter()
    t_init = t1-t0

    rows.append([0, Sim.numAgents, seed, Sim.avg_utility(Sim.Repo[0]), smax, smin, t_init, 0, Sim.numDesigns, Sim.utility_parameter, Sim.numSameDesigns])
   
   #run phase 1 the model for different iterations per agent:
    for i in range(1,maxItsPerA): #i = iterations per agent
        t0 = time.perf_counter()
        run_phase1(Sim, Sim.numAgents)
        t1 = time.perf_counter()
        t_phase1 = t1-t0

        # get the best average utility in the repository
        s_best = max([Sim.Uavg[Sim.Repo[i]] for i in range(len(Sim.Repo))])

        rows.append([i, Sim.numAgents, seed, s_best, smax, smin, t_init, t_phase1, Sim.numDesigns, Sim.utility_parameter, Sim.numSameDesigns])
    
    return rows

#main function
if __name__ == "__main__":

#input: filename
    if len(sys.argv) != 2:
        raise ValueError('Name of output file is missing')
    filename = sys.argv[1] + str('.csv')
    print("output filename = " + str(filename))

    iteration_tuples = product(agentNums, seeds) #iter_tuple = (iterations per agent, numAgents)
    print(iteration_tuples)


    p = multiprocessing.Pool() 
    result = p.map(runModel_timePerAgent, iteration_tuples) 

    # #serial version:
    # result = []
    # for t in iteration_tuples:
    #     result.append(runModel_timePerAgent(t))
    
    #write results to a csv file:
    print("writing data to csv file...")

    pathtofile = Path.cwd()/'data'/'final'/filename
     # writing to csv file  
    with open(pathtofile, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        # writing the fields  
        fields = ['iterations per agent', 'number of agents', 'random seed', 'average utility of best design', 'smax', 'smin', 't_init','t_phase1', 'number of created designs', 'utility param', 'number of same designs']

        csvwriter.writerow(fields)  
            
        # writing the data rows  
        for row in result:
            csvwriter.writerows(row) 

