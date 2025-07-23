from model.run import *
import sys
import time
import multiprocessing 
import os 
from itertools import product
import csv
from pathlib import Path

''' This script runs phase 1 and phase 2 of the model for different numbers of the agents over several repeats 

Input:
    name of the file, where the results will be stored

The results will be stored in filename.csv in the directory ./data/final

'''

nAgents = np.geomspace(10,10000, num=30, dtype=int)
seeds = np.linspace(1,100, num=100, dtype=int)

#Model Parameters:
votesPerDesign = 10
stepsPerAgent = 2
div = 0.5
utility_shape = 500
util_smoothness = 30


def runModel_numAgents(random_seed):
    '''
    This function runs phase 1 and 2 with different numbers of agents from the list nAgents

    Input: 
        random_seed: seed for the random generator

    Returns:
        List with each row containing the model parameters and the corresponding scores of the final design '''

    
    print("Worker process id for {0}: {1}".format(random_seed, os.getpid()))

    rows = []

    # initialize the model
    Sim = Simulation(numAgents = nAgents[0], diversity=div, seed=random_seed, utility_parameter=util_smoothness, u_shape=utility_shape)
    
    for i in range(len(nAgents)):
        if i != 0:
            # add agents to the model and reset the repository to one initial design
            Sim.add_agents(nAgents[i]-nAgents[i-1])
            Sim.initialize_Repo()
        print(Sim.numAgents)

        # running phase 1
        run_phase1(Sim, steps=stepsPerAgent*Sim.numAgents)

        # running both steps of phase 2
        x_voted, s_voted = run_phase2(Sim)

        # get the max and min average utility
        x_max = Sim.find_maximum_design(Sim.Uavg)
        x_min = Sim.Repo[0]
        smax = Sim.Uavg[x_max]
        smin = Sim.Uavg[x_min]

        # compute the best and median average utility of the designs in the repository
        s_best = max([Sim.Uavg[Sim.Repo[i]] for i in range(len(Sim.Repo))])
        s_med = np.median([Sim.Uavg[Sim.Repo[i]] for i in range(len(Sim.Repo))])

        rows.append([Sim.numAgents, random_seed, s_best, s_voted, smax, smin, Sim.numDesigns, Sim.utility_parameter, Sim.numSameDesigns, Sim.diversity, s_med])

    return rows
        


#main function
if __name__ == "__main__":

#input: filename
    if len(sys.argv) != 2:
        raise ValueError('Name of output file is missing')
    filename = sys.argv[1] + str('.csv')
    print("output filename = " + str(filename))


    t0 = time.perf_counter()

    p = multiprocessing.Pool() 
    result = p.map(runModel_numAgents, seeds) 

    t1 = time.perf_counter()

    print('Runtime[sec]: ', t1-t0)

    # #serial  version:
    # t0 = time.perf_counter()
    # result = []
    # for s in seeds:
    #         result.append(runModel_numAgents(s))
    # t1 = time.perf_counter()

    # print('Runtime[sec] serial: ', t1-t0)

    
    #write results to a csv file:
    print("writing data to csv file...")
    pathtofile = Path.cwd()/'data'/'final'/filename
     # writing to csv file  
    with open(pathtofile, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  

        # writing the fields  
        fields = ['number of agents', 'random seed', 'average utility of best design', 'average utility of voted design','smax', 'smin', 'number of created designs', 'utility param', 'number of same designs', 'diversity param', 'median']

        csvwriter.writerow(fields)  
            
        # writing the data rows  
        for row in result:
            csvwriter.writerows(row) 

