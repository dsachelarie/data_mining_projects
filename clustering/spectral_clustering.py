import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

"""
Implementation of spectral clustering.
"""

with open("./cowdist.csv", 'r') as file:
    # Build matrix of similarities between cows and display the graph
    rows = list(csv.reader(file))[1:]
    similarities = np.zeros((6, 6))
    row_id = 0

    for i in range(6):
        for j in range(i + 1, 6):
            similarities[i][j] = 1 - float(rows[row_id][2])  # Convert distance to similarity
            similarities[j][i] = similarities[i][j]
            row_id += 1

    rows, cols = np.where(similarities > 0)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500)
    plt.show()

    # Calculate Laplacian matrix
    degree = np.zeros((6, 6))

    for i in range(6):
        degree[i][i] = np.sum(similarities[i])

    laplacian = degree - similarities

    # Get eigenvector and plot the resulting 1D data
    eigen_result = np.linalg.eig(laplacian)
    eigenvector = eigen_result[1][1]  # Get second smallest eigenvector

    plt.plot(eigenvector, np.zeros_like(eigenvector), "x")
    plt.show()

    # Same as above, but using the random-walk Laplacian
    laplacian = np.matmul(np.linalg.inv(degree), laplacian)
    eigen_result = np.linalg.eig(laplacian)
    eigenvector = eigen_result[1][1]

    plt.plot(eigenvector, np.zeros_like(eigenvector), "x")
    plt.show()
