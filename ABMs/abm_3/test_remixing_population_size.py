import numpy as np

import deep_seek_remix


def make_filename(max_N, design_dim, T, S, M, K):
    return f"results_N{max_N}_{design_dim}d_T{T}_S{S}_M{M}_K{K}"


if __name__ == "__main__":
    import pickle
    from tqdm.auto import tqdm

    # Parameter setup for heterogeneous agents
    max_N = 10000  # Population size

    agent_params = [{
        'sigma': np.random.lognormal(0.5, 1),
        'temp': np.random.lognormal(-1, 0.3),
        'mutate_rate': np.random.beta(5, 1),
        'mutate_strength': np.random.lognormal(0, 0.1)
    } for _ in range(max_N)]

    results = []

    design_dim = 2  # d-dimensional design space
    T = 1  # iterations
    S = 10 # designs per menu
    M = 1  # votes per agent
    K = 3  # Top designs in final vote

    repetitions = 100

    for N in tqdm(np.round(np.logspace(1, 4, 12), -1)):
        for repetition in range(repetitions):
            N = int(N)

            reduced_params = agent_params[0:N]

            # Run simulation
            final_design, utility, max_utility = deep_seek_remix.simulate_population(
                agent_params=reduced_params,
                design_dim=design_dim,
                T=T,
                S=S,
                M=M,
                K=K
            )

            print("number of agents:", N)
            print("Final selected design:", final_design)
            print("Final selected utility:", utility)

            results.append({
                "N": N,
                "repetition": repetition,
                "design": final_design,
                "utility": utility,
                "max": max_utility,
            })

    filename = make_filename(max_N, design_dim, T, S, M, K)

    with open(f"{filename}_{repetitions}.pkl", "wb") as file:
        pickle.dump(results, file)
