from model.run import *
from model.simulation import *

import sys
import time
import multiprocessing 
import os 
from itertools import product
from pathlib import Path
import csv

'''
This script runs phase 2 for different votes per solution in the first step of phase 2 
Input:
    name of the file, where the results will be stored

The results will be stored in filename.csv the directory ./data/final
'''

votesPerDesign = np.geomspace(1, 100, num=20, dtype=int)
seeds = np.linspace(1,100,num=100,dtype=int)

# Model parameters
div = 0.5
smoothness = 30
numOfAgents = [100]
numDesigns = [100,1000]
stepsPerAgent = 2

# if random_repo == False, phase 2 is evaluated on the repository after running phase 1 (see voting_on_repository_after_phase1)
# if random_repo == True, phase 2 is evaluated on designs, that are uniformly distributed over the solution space
random_repo = False 

def voting_on_uniform_distr(iter_tuple): 
    '''
    evaluates phase 2 for different votes per designs on uniformly distributed designs

    Input: 
        iter_tuple = (number of agents, seed for the random generator)

    Returns:
        List with each row containing the model parameters and the corresponding scores of the final design '''

    numAgents = iter_tuple[0]
    random_seed = iter_tuple[1]
    numDesigns = iter_tuple[2]

    print("Worker process id for {0}: {1}".format(iter_tuple, os.getpid()))

    rows = []
    Sim = Simulation(numAgents, seed=random_seed, diversity=div, utility_parameter=smoothness)
    x_max = Sim.find_maximum_design(Sim.Uavg)
    x_min = Sim.Repo[0]
    smax = Sim.Uavg[x_max]
    smin = Sim.Uavg[x_min]

    #generate uniformly distributed designs:
    for i in range(numDesigns):
        x = Sim.rng.integers(0, Sim.utilShape[0])
        y = Sim.rng.integers(0, Sim.utilShape[1])
        Sim.Repo.append((x,y))
    Sim.numDesigns = len(Sim.Repo)

    s_best = max([Sim.Uavg[Sim.Repo[i]] for i in range(len(Sim.Repo))])
    s_median = np.median([Sim.Uavg[Sim.Repo[i]] for i in range(len(Sim.Repo))])
    
    for k in votesPerDesign:
        for r in range(10):
            #stage 1 only:
            best10_ids, best_id = voting_step1(Sim, k)
            sv1 = Sim.avg_utility(Sim.Repo[best_id])
            #stage 1+2:
            x2, sv12 = voting_step2(Sim, best10_ids)

            rows.append([Sim.numAgents, random_seed, k, s_best, sv1, sv12, smax, smin, Sim.numDesigns, Sim.utility_parameter, Sim.numSameDesigns,0, s_median])

    return rows
        


def voting_on_repository_after_phase1(iter_tuple): # iter_tuple = (numAgents, random_seed)
    '''
    This function evaluates phase 2 on the created designs after running phase 1 of the model.
    Algorithm:
        1. Initialize the model and run phase 1.
        2. For k in votesPerDesign:
            Run the first step of phase 2. Every solution is assigned to  k agents. 
            Store the solution with the best score.
            Run the second step of phase 2 on the 10 best solutions from the previous step and store the solution with the best score.

    Input: 
        iter_tuple = (number of agents, seed for the random generator)

    Returns:
        List with each row containing the model parameters and the corresponding scores of the final design
    '''

    numAgents = iter_tuple[0]
    random_seed = iter_tuple[1]
    print("Worker process id for {0}: {1}".format(iter_tuple, os.getpid()))

    rows = []

    # initialize the model and run phase 1
    steps = 2*numAgents
    Sim = Simulation(numAgents, seed=random_seed, diversity=div, utility_parameter=smoothness)
    run_phase1(Sim, steps)

    # get designs with best/worst average utilities
    x_max = Sim.find_maximum_design(Sim.Uavg)
    x_min = Sim.Repo[0]
    smax = Sim.Uavg[x_max]
    smin = Sim.Uavg[x_min]

    # get the best/median average utility from the repository (out of all created designs)
    s_best = max([Sim.Uavg[Sim.Repo[i]] for i in range(len(Sim.Repo))])
    s_median = np.median([Sim.Uavg[Sim.Repo[i]] for i in range(len(Sim.Repo))])

    # run phase 2: voting on all created designs in the repository
    for k in votesPerDesign:
        #for r in range(10):
        #run step 1:
        best10_ids, best_id = voting_step1(Sim, k)
        sv1 = Sim.avg_utility(Sim.Repo[best_id]) # average utility of the design that has the best score after step 1
        #run step 2:
        x2, sv12 = voting_step2(Sim, best10_ids) # average utility of the design that has the best score after step 1 and 2

        rows.append([Sim.numAgents, random_seed, k, s_best, sv1, sv12, smax, smin, Sim.numDesigns, Sim.utility_parameter, Sim.numSameDesigns, steps, s_median])

    return rows
        

#main function
if __name__ == "__main__":
    ''' 
    Input:
        name of the file, where the results will be stored
    
    The results will be stored in filename.csv the directory final_data
    '''

#input: filename where the results are stored
    if len(sys.argv) != 2:
        raise ValueError('Name of output file is missing')
    filename = sys.argv[1] + str('.csv')
    print("output filename = " + str(filename))


    t0 = time.perf_counter()
    # creating a pool
    p = multiprocessing.Pool() 

    result = []

    if random_repo == True:
        iteration_tuples_randomRepo = product(numOfAgents, seeds, numDesigns)
        result = p.map(voting_on_uniform_distr, iteration_tuples_randomRepo) 

    else:
        iteration_tuples = product(numOfAgents, seeds) #iter_tuple = (iterations per agent, numAgents)
        result = p.map(voting_on_repository_after_phase1, iteration_tuples) 


    t1 = time.perf_counter()
    print('Runtime[sec]: ', t1-t0)

 
    print("writing data to csv file...")

    pathtofile = Path.cwd()/'data'/'final'/filename
     # writing to csv file  
    with open(pathtofile, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  

        # writing the fields  
        fields = ['number of agents', 'random seed', 'votes per solution', 'average utility of best design', 'average utility of final design (stage 1)','average utility of final design (stage 1+2)','smax', 'smin', 'number of created designs', 'utility param', 'number of same designs', 'iterations of phase 1', 'median of avg utility in repo']

        csvwriter.writerow(fields)  
            
        # writing the data rows  
        for row in result:
            csvwriter.writerows(row) 

