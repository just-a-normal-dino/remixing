import numpy as np
from collections import defaultdict
import tqdm


class Agent:
    def __init__(self, ideal_point, sigma, mutate_rate, mutate_strength):
        self.ideal_point = np.array(ideal_point)
        self.sigma = sigma  # Utility function breadth
        self.mutate_rate = mutate_rate  # Mutation probability per dimension
        self.mutate_strength = mutate_strength  # Mutation magnitude

    def utility(self, design):
        """Gaussian utility based on distance from ideal point"""
        distance = np.linalg.norm(design - self.ideal_point)
        return np.exp(-distance ** 2 / (2 * self.sigma ** 2))

    def select_design(self, pool):
        """Uniform random selection """
        return pool[np.random.choice(len(pool))]

    def evolve(self, design):
        """Mutate each dimension with probability mutate_rate"""
        mutated = design.copy()
        mask = np.random.rand(*design.shape) < self.mutate_rate
        mutated += mask * np.random.normal(0, self.mutate_strength, design.shape)
        return mutated

    def accept(self, original, evolved):
        """Accept evolved design if utility increases"""
        return self.utility(evolved) > self.utility(original)


def first_stage(agents, init_designs, steps):
    """Simulate design creation process"""
    pool = [np.array(d) for d in init_designs]
    # n_random_agents = steps/2 * len(agents)
    # random_sample_agents = np.random.choice(agents, )
    for _ in tqdm.tqdm(range(steps)):
        for agent in agents:
            selected = agent.select_design(pool)
            evolved = agent.evolve(selected)
            if agent.accept(selected, evolved):
                pool.append(evolved)
    return pool


def get_all_utilities(pool, agents):
    utilities = np.zeros(len(pool))
    for i, design in enumerate(pool):
        utilities[i] = np.mean([agent.utility(design) for agent in agents])
    return utilities


def second_stage(agents, designs, sample_size, num_samples, top_k):
    """Simulate voting process"""
    # First voting phase
    vote_counts = defaultdict(int)
    scores = defaultdict(float)
    scored_designs = []

    for agent in tqdm.tqdm(agents):
        for _ in range(num_samples):
            indices = np.random.choice(len(designs), sample_size, replace=True)
            sample = [designs[i] for i in indices]
            utilities = [agent.utility(d) for d in sample]
            for index, utility in zip(indices, utilities):
                if index in scores:
                    scores[index] = (scores[index] + utility)/2
                else:
                    scores[index] = utility

    # Select top K designs
    top_k_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    finalists = [designs[i] for i, _ in top_k_indices]

    # Final vote
    final_votes = np.zeros((top_k, len(agents)))
    for i,agent in enumerate(agents):
        final_votes[:,i] = np.array([agent.utility(d) for d in finalists])

    mean_votes = np.mean(final_votes, axis=1)
    selected = np.argmax(mean_votes)

    return finalists[selected]


def simulate_population(agent_params, design_dim, T, S, M, K):
    """Full simulation with parameterized population"""
    # Create agents with specified parameters
    agents = [
        Agent(
            ideal_point=np.random.randn(design_dim),  # Sample from normal distribution
            sigma=params['sigma'],
            mutate_rate=params['mutate_rate'],
            mutate_strength=params['mutate_strength']
        )
        for params in agent_params
    ]

    # Initial designs (random seeds)
    # init_designs = [np.random.randn(design_dim) for _ in range(2)]
    init_designs = [np.array([-10,-10]).astype(float), np.array([10, 10]).astype(float)]

    # Run both stages
    pool = first_stage(agents, init_designs, T)
    winner = second_stage(agents, pool, S, M, K)
    # utilities = get_all_utilities(pool, agents)

    # get the utility of the winner
    utility = np.mean(np.array([agent.utility(winner) for agent in agents]))
    # max_found = np.max(utilities)
    max_utility = np.mean([agent.utility(np.zeros(design_dim)) for agent in agents])

    return winner, utility, max_utility


def get_design_space(agent_params, design_dim):
    # Create agents with specified parameters
    agents = [
        Agent(
            ideal_point=np.random.randn(design_dim),  # Sample from normal distribution
            sigma=params['sigma'],
            mutate_rate=params['mutate_rate'],
            mutate_strength=params['mutate_strength']
        )
        for params in agent_params
    ]

    agent_points = np.array([agent.ideal_point for agent in agents])

    x = np.linspace(-3, 3, 21)
    y = np.linspace(-3, 3, 21)
    # z = np.linspace(-10, 10, 21)

    X, Y = np.meshgrid(x, y, indexing='ij')
    grid_shape = X.shape

    # Flatten the meshgrid for evaluation
    points = np.stack([X.ravel(), Y.ravel()], axis=1)

    # get all the distances
    # distances = [[np.linalg.norm(point - agent_points[i]) for i in range(len(agent_points))] for point in points]
    # print(distances)

    # Initialize utility array: (num_agents, *grid_shape)
    utilities = np.zeros((len(agents), *grid_shape))

    for i, agent in enumerate(agents):
        # Compute utility at each point for this agent
        agent_utilities = np.array([agent.utility(point) for point  in points])
        utilities[i] = agent_utilities.reshape(grid_shape)

    return X, Y, utilities  # utilities.shape = (num_agents, 101, 101)


# Example usage
if __name__ == "__main__":
    # Pa rameter setup for heterogeneous agents
    N = 100000 # Population size
    design_dim = 2  # d-dimensional design space
    agent_params = [{
        'sigma': np.random.lognormal(0.5, 1),
        'temp': np.random.lognormal(-1, 0.3),
        'mutate_rate': np.random.beta(5, 1),
        'mutate_strength': np.random.lognormal(0, 0.5)
    } for _ in range(N)]

    # get design space
    # X, Y, utilities = get_design_space(agent_params, design_dim)
    # utilities = utilities.mean(axis=0)
    # print(utilities)
    # print(utilities.max())
    # print(utilities.min())

    import matplotlib.pyplot as plt

    # plt.imshow(utilities)
    plt.show()
    # Run simulation
    final_design, utility, max_utility = simulate_population(
        agent_params=agent_params,
        design_dim=design_dim,
        T=1,  # iterations
        S=10,  # designs per menu
        M=1,  # votes per agent
        K=3  # Top k designs in final vote
    )

    print("Final selected design:", final_design)
    print("final design utility:", utility)
    print("max possible utility:", max_utility)
