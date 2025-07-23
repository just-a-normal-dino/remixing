from .simulation import *

def run_phase1(model, steps, random_mutation=True):
    '''
    Phase 1: Creating solutions collectively
    This function excutes the following steps repeatedly:
        - Select one agent a randomly.
        - The selected agent is exposed to 10 random solutions from the repository. 
        Out of these 10 solutions, the agent selects the solution that maximizes the utility of the agent. 
        If less than 10 are available, the agent selects the best solution out of the whole repository.
        - The agent creates a new solution by “mutating” the selected solution. 
        Practically, this means that the new solution is chosen randomly out of the 8 neighbors of the selected solution.
        - The agent includes the new solution in the repository if its utility is greater or equal than the agent's utility of the selected solution
    Parameters:
    - steps, number of repetitions of the above steps
    - model, model that contains the agents, teir utility functions and the repository
    '''
    for it in range(steps):
        availableDesignIds = model.random_designs(model.numExposed)

        ### select 1 agent randomly ###
        a = model.rng.integers(model.numAgents)
        ### agent selects design ###
        selectedId = model.selectDesign_best(a, availableDesignIds)
        x_selected = model.Repo[selectedId]
        ### agent creates new design with mutation ###
        x_new = model.mutate_discrete(x_selected)

        # agent checks if he wants to include the new design to the repo and if so, he includes it
        if model.utilities[a][x_new] >= model.utilities[a][x_selected]:
            model.include_design(x_new)


def run_phase2(model, k=10, bothsteps = True):
    '''
    Phase 2: The agents choose one final solution through voting
    step 1: Every solution in the repository is randomly assigned to k agents, that vote on this solution. 
	step 2: All agents vote on the 10 solutions with best total scores. The final solution is the solution, that has the highest total score after this step.
    Parameters:
    - k, solutions per agent in step q
    - bothsteps, True exectues steps 1 and 2
                 False executes step 1 only and returns the solution with the best score after step 1.
    '''
    best_design_ids, id_final_design = voting_step1(model, k)
    if bothsteps == True: #steps 1 + 2
        final_design, uavg = voting_step2(model, best_design_ids)
        return final_design, uavg #final design, avg utility of final design
    else: # step 1 only: best design after step 1, avg utility of best design after step 1
        return model.Repo[id_final_design], model.avg_utility(model.Repo[id_final_design])


def voting_step1(model, k):
    '''every design in the repository is randomly assigned to k agents, that vote on that design. score of agent a for design x = utility_a(x)
    2. select 10 designs with highest total scores. Total score of design = average over scores from all agents, that voted on that design'''
    N = len(model.Repo)

    votes = np.zeros(N)

    best_design_ids = []

    if N > 10:
    ### step 1: every design is evaluated by k random agents: in average every agent votes for k * (numDesigns / numAgents) designs
        for x_id in range(N):
            aIds = model.rng.choice(model.numAgents, size=k)
            for a in aIds:
                votes[x_id] += model.utility(model.Repo[x_id], a)

        votes /= k #take the average
        #get k designs with best votes:
        best_design_ids = np.argsort(votes)[N-10:N]
        id_best_design = best_design_ids[-1]
        assert(votes[id_best_design] == max(votes))
        return best_design_ids, id_best_design

    else: #skip step 1 if N <= 10
        best_design_ids = np.linspace(0, N-1, N, dtype=int)
        return best_design_ids, np.argmax([model.avg_utility(x) for x in model.Repo])

def voting_step2(model, best_design_ids):
    '''all agents vote on the designs that are in the best_design_ids
       returns the best average utility over the input designs'''
    designs = [model.Repo[xid] for xid in best_design_ids]
    avgutils = [model.avg_utility(x) for x in designs]
    bestDesign = designs[np.argmax(avgutils)]
    assert(model.avg_utility(bestDesign) == max(avgutils))

    return bestDesign, model.avg_utility(bestDesign)
