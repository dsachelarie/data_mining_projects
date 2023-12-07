import random
import sys
import numpy as np
from plotting import plot_contrasts, plot_averages, plot_variance

"""
An exploration of the "curse of dimensionality": the extent to which the number of dimensions affects the robustness of distance measures.
The effect is assessed for Lp norms with various p values, and distances are calculated from origin.
Findings: 
- no matter the norm used, for high dimensions, contrast disappears;
- the maximum, minimum, and average distances increase faster for lower p's;
- the average variance increases for low p's and decreases for higher p's.
"""

def calculate_norm(p: float, dataset: [], k: int):
    max_distance = 0
    min_distance = sys.maxsize
    sum = 0
    distances = []

    for point in dataset:
        point = point[:k]

        if p == -1:
            distance = max(point)

        else:
            distance = 0

            for dimension in point:
                distance += np.abs(dimension)**p

            distance = distance**(1/p)

        if distance > max_distance:
            max_distance = distance

        if distance < min_distance:
            min_distance = distance

        sum += distance
        distances.append(distance)

    variance = 0
    mean = sum / len(dataset)

    for sample in distances:
        variance += ((sample - mean)**2)

    variance /= (len(dataset) - 1)

    return {
        "p": p,
        "dimensionality": k,
        "max_distance": max_distance,
        "min_distance": min_distance,
        "mean_distance": mean,
        "var_distance": variance,
        "contrast": (max_distance - min_distance) / min_distance
    }


# Create a collection of datasets, each with n points of dimensionality k
datasets = []

random.seed(10)
q = 100  # number of datasets
n = 100  # number of points
k = 100  # max size of dimensionality

for i in range(q):
    dataset = []

    for j in range(n):
        all_zeros = True
        point = []

        # Add non-zero point of k dimensions
        while all_zeros:
            point = []

            for l in range(k):
                point.append(random.random())

            for l in range(k):
                if point[l] != 0:
                    all_zeros = False
                    break

        dataset.append(point)

    datasets.append(dataset)

# Calculate distance to all points from the origin using a selection of dimensionalities and Lp norms;
# Plot average minimum and maximum distance, average variance of distance, and relative constrasts.
dimensionalities = [2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
p_values = [-1, 0.5, 1, 2, 5]  # -1 stands for infinity
results = []

for k in dimensionalities:
    for dataset in datasets:
        for p in p_values:
            results.append(calculate_norm(p, dataset, k))

plot_contrasts(results, p_values, dimensionalities)
plot_averages(results, p_values, dimensionalities)
plot_variance(results, p_values, dimensionalities)
