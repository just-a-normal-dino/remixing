import matplotlib.pyplot as plt
import pickle

import numpy as np
from test_remixing_population_size import make_filename

N = 10000
design_dim = 2  # d-dimensional design space
T = 1
S = 10
M = 1
K = 3
filename = make_filename(N, design_dim, T, S, M, K)

with open(f"{filename}.pkl", "rb") as file:
    data = pickle.load(file)

print(data)

pop_sizes = np.round(np.logspace(2, 4, 10), -2)

plt.plot(pop_sizes, [data[n]["utility"] for n in pop_sizes])
plt.xlabel("population size")
plt.ylabel("final voted design utility")
plt.savefig(f"plot_{filename}.png")
plt.show()
