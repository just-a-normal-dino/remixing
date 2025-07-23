import random
import numpy as np
from scipy.ndimage import gaussian_filter

class Simulation():
    '''Implementation of the model'''
    
    def __init__(self, numAgents, utility_parameter = 30, u_shape = 500, numInitDesigns=1, numExposed=10, diversity=0.5, seed=None):
        '''  
        Parameters:

            numAgents: number of agents 

            utility_parameter: parameter for determining the “smoothness” of the utility functions. The bigger, the smoother the utility functions.
            
            u_shape: number of gridpoints in each dimension
            
            numInitDesigns: number of initial designs. 
                            If numInitDesigns == 1, the initial design is the design with the worst average utility.
                            If numInitDesigns > 1, the initial designs are created randomly.
            
            numExposed: number of designs, from which an agent can select in phase 1
            
            diversity: parameter for determining the noise in the utility functions. The bigger, the more the utility functions differ between agents
            
            seed: seed for the random generator 
    '''
        
        self.rng = np.random.default_rng(seed)

        #create the utilities of the agents:
        self.utilShape = (u_shape, u_shape)
        self.numAgents = numAgents
        self.utility_parameter = utility_parameter
        self.diversity = diversity 
        self.utilities = []
        self.U0 = self.create_utility(self.utility_parameter)
        '''create utilities of the agents:
            utility of agent a = (1-diversity) * U0 + diversity * random utility'''
        for i in range(numAgents):
            u = self.create_utility(self.utility_parameter) 
            self.utilities.append(self.resize(self.diversity * u + (1-self.diversity) * self.U0))

        #compute the average utility over all agents:
        self.Uavg = sum(self.utilities) / self.numAgents

        #create Repository and initialize first designs
        self.numDesigns = 0
        self.numInitDesigns = numInitDesigns

        self.Repo = []
        self.initialize_Repo()

        #parameters for creating designs:
        self.numExposed = numExposed #from how many designs agents can select

    def utility(self, x, agent):
        ''' returns the utility of agent a on the design x'''
        return self.utilities[agent][x]
    
    def avg_utility(self, x):
        ''' returns the average utility of the design x'''
        return self.Uavg[x]

    def create_utility(self, blur_parameter):
        ''' creates utility function from random noise'''
        noise = self.rng.random(self.utilShape)
        blurred_image = gaussian_filter(noise, sigma=blur_parameter)
        U = self.resize(blurred_image)
        return U
    
    def resize(self, image): # resize between 0 and 1
        i3 = image-np.min(image)
        i4 = i3/np.max(i3) 
        return i4

    def add_agents(self, n):
        ''' adds n agents to the model '''
        Nold = self.numAgents
        self.numAgents += n
        self.Uavg = self.Uavg * Nold
        for i in range(n):
            #create a new utility function for the agent:
            u_tmp = self.create_utility(self.utility_parameter)
            u = self.resize(self.diversity * u_tmp + (1-self.diversity) * self.U0)
            self.utilities.append(u)
            self.Uavg += u
    
        #update the average utility:
        self.Uavg /= self.numAgents
        assert(len(self.utilities) == self.numAgents)

    def initialize_Repo(self):
        '''initialized the first designs in the repository'''
        self.numDesigns = 0
        if self.numInitDesigns == 1: # x_init = design with minimal average utlity
            design = np.unravel_index(np.argmin(self.Uavg, axis=None), self.utilShape)
            self.Repo = [design]
            self.numDesigns += 1
        else:
            #initialize designs randomly
            for i in range(self.numInitDesigns):
                x_random = self.create_random_design()
                self.Repo.append(x_random)
                self.numDesigns += 1
        assert(len(self.Repo) == self.numDesigns)

    def create_random_design(self):
        x1 = self.rng.integers(0, self.utilShape[0])
        x2 = self.rng.integers(0, self.utilShape[1])
        return (x1, x2)
    
    def is_valid_in_dim(self, value, d):
        '''value = value of the design in d-th dimension
            returns true if value is inside the design space'''
        return (value < self.utilShape[d] and value >= 0)
    
    def is_valid(self, x):
        ''' returns true if the design x is inside the design space'''
        return (0 <= x[0] < self.utilShape[0] and 0 <= x[1] < self.utilShape[1])
    
    def isEqual(self, design1, design2):
        ''' returns True if design 1 is equal to design 2'''
        return (np.linalg.norm(design1 - design2) <= self.tol)

    def newest_designs(self, k):
        '''returns the id's of the k newest designs in the Repo: [numDesigns-k, numDesigns-k+1, ..., numDesigns-1]'''
        N = self.numDesigns
        return [i for i in range(max(0,N-k), N)]
      
    def random_designs(self, k):
        '''returns the id's of k random designs in the Repo'''
        if k < self.numDesigns:
            return self.rng.choice(self.numDesigns, k)
        else:
            return np.arange(self.numDesigns)

    def selectDesign_best(self, agent, availableIds):
        ''' availableIds: indices to designs in the repository
            returns the index of the best design from availableIds
        '''
        u = lambda x_id: self.utility(self.Repo[x_id], agent)
        return max(availableIds, key=u)
    
    def bestDesign(self, agent, availableDesigns):
        '''returns the design out of availableDesigns, that maximized the average utility.
        '''
        u = lambda x: self.utility(x, agent)
        return max(availableDesigns, key=u)

    def neighbour_designs(self, design):
        ''' returns the 8 neighbour designs of the design '''
        neighbours = []
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                if not (i == 0 and j == 0):
                    x = (design[0]+i, design[1]+j)
                    if self.is_valid(x):
                        neighbours.append(x)
        return neighbours

    def mutate_discrete(self, design):
        ''' returns one random design out of the neighbours of the given design '''
        neighbours = self.neighbour_designs(design)
        return tuple(self.rng.choice(neighbours))
    
    def find_maximum_design(self,utility):
        ''' 
        utility: 2D image of an utility function

        returns the maximal value of utility'''
        max_position = np.unravel_index(np.argmax(utility), utility.shape)
        return max_position



    def include_design(self, x_new):
        '''includes the design in the repository, if it isn't already in the repo'''
        if x_new not in self.Repo:
            self.Repo.append(x_new)
            self.numDesigns += 1
   

     
  
    