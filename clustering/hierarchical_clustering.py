import sys
import numpy as np
import csv

"""
Implementation of hierarchical clustering for different values of K using single, complete, and average linkage metrics.
"""

def complete_linkage(cluster1: [], cluster2: [], distances: []) -> float:
    max_dist = 0

    for el1 in cluster1:
        for el2 in cluster2:
            if distances[el1][el2] > max_dist or distances[el2][el1] > max_dist:
                max_dist = max(distances[el1][el2], distances[el2][el1])  # Only the half of the matrix above the main diagonal is used, the other has 0's

    return max_dist


def single_linkage(cluster1: [], cluster2: [], distances: []) -> float:
    min_dist = sys.maxsize

    for el1 in cluster1:
        for el2 in cluster2:
            if (distances[el2][el1] == 0 and distances[el1][el2] < min_dist) or (distances[el1][el2] == 0 and distances[el2][el1] < min_dist):
                min_dist = max(distances[el1][el2], distances[el2][el1])

    return min_dist


def average_linkage(cluster1: [], cluster2: [], distances: []) -> float:
    dist_sum = 0

    for el1 in cluster1:
        for el2 in cluster2:
            dist_sum += max(distances[el1][el2], distances[el2][el1])

    return dist_sum / len(cluster1) / len(cluster2)


with open("./birdspecies.csv", 'r') as file:
    # Feature extraction
    rows = list(csv.reader(file, delimiter=";"))[1:]
    bmis = []
    wsis = []

    for row in rows:
        length = [int(x) for x in row[2].split("-")]
        weight = [int(x) for x in row[4].split("-")]
        wing_span = [int(x) for x in row[3].split("-")]
        avg_length = sum(length) / len(length)
        avg_weight = sum(weight) / len(weight)
        avg_wing_span = sum(wing_span) / len(wing_span)

        bmis.append(avg_weight / (avg_length**2))
        wsis.append(avg_wing_span / avg_length)

    # Calculate pairwise Euclidean and overlap distances and combine them
    n = len(bmis)
    euclidean = np.zeros((n, n))
    overlap = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            euclidean[i][j] = np.sqrt((bmis[i] - bmis[j])**2 + (wsis[i] - wsis[j])**2)
            overlap[i][j] = 1 - 0.5 * ((1 if rows[i][5] == rows[j][5] else 0) + (1 if rows[i][6] == rows[j][6] else 0))

    flattened_euclidean = []
    flattened_overlap = []
    combined = np.zeros((n, n))
    lmbda = 0.6

    for i in range(n):
        for j in range(i + 1, n):
            flattened_euclidean.append(euclidean[i][j])
            flattened_overlap.append(overlap[i][j])

    std_euclidean = np.std(flattened_euclidean)
    std_overlap = np.std(flattened_overlap)

    for i in range(n):
        for j in range(i + 1, n):
            combined[i][j] = lmbda * euclidean[i][j] / std_euclidean + (1 - lmbda) * overlap[i][j] / std_overlap

    # Perform hierarchical clustering using complete, single, and average linkage, for different values of K
    ks = [5, 10, 15, 20, 25]
    results = []

    for k in ks:
        for linkage in {"single", "complete", "average"}:
            clusters = []

            for i in range(n):
                clusters.append([i])

            while len(clusters) > k:
                i_min = -1
                j_min = -1
                min_dist = sys.maxsize

                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        if linkage == "single":
                            dist = single_linkage(clusters[i], clusters[j], combined)
                        elif linkage == "complete":
                            dist = complete_linkage(clusters[i], clusters[j], combined)
                        else:
                            dist = average_linkage(clusters[i], clusters[j], combined)

                        if dist < min_dist:
                            min_dist = dist
                            i_min = i
                            j_min = j

                clusters.append(clusters[i_min] + clusters[j_min])

                del clusters[i_min]
                del clusters[j_min - 1]

            results.append({"k": k, "clustering": clusters.copy(), "linkage": linkage})

    print(results)
