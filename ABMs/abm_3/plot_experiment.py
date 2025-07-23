import matplotlib.pyplot as plt
import pickle
import pandas as pd

import numpy as np
from test_remixing_population_size import make_filename

N = 10000
design_dim = 2  # d-dimensional design space
T = 1
S = 10
M = 1
K = 3
filename = make_filename(N, design_dim, T, S, M, K)

with open(f"{filename}_{100}.pkl", "rb") as file:
    data = pickle.load(file)

print(data)

pop_sizes = np.round(np.logspace(2, 4, 10), -2)

df = pd.DataFrame(data=data)

print(df)

# means = df.groupby(by=["N"]).mean()
# stds = df.groupby(by=["N"]).std()

# Calculate the ratio for each individual data point
df['utility_ratio'] = df['utility'] / df['max']

# Calculate the mean and standard deviation of the utility ratio grouped by N
mean_utility_ratio = df.groupby(by=["N"])['utility_ratio'].mean()
std_utility_ratio = df.groupby(by=["N"])['utility_ratio'].std()

# Plot the utility ratio mean with fill_between for standard deviation
plt.figure(figsize=(10, 6))
plt.plot(mean_utility_ratio.index, mean_utility_ratio, color='blue', label='Mean Utility / Max Ratio')
plt.fill_between(mean_utility_ratio.index,
                 mean_utility_ratio - std_utility_ratio,
                 mean_utility_ratio + std_utility_ratio,
                 color='lightblue', alpha=0.5, label='$\pm$1 Standard Deviation')
plt.xscale("log")
plt.xlabel('N')
plt.ylabel('Utility / Max Utility')
plt.title('Utility / Max Ratio as a Function of N with $\pm$1 Std Dev Shaded Area')
plt.grid(True)
plt.legend()
plt.show()
# plt.plot(pop_sizes, [data[n]["utility"] for n in pop_sizes])
# plt.xlabel("population size")
# plt.ylabel("final voted design utility")
# plt.savefig(f"plot_{filename}.png")
# plt.show()

df.to_csv(f"{filename}_deepseekmodel.csv")
