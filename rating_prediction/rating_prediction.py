import numpy as np

"""
Rating prediction using neighborhood-based collaborative filtering.
"""

# Table of movie ratings per user
table = [[3, 1, 2, 2, 0, 2], [4, 2, 3, 3, 4, 2], [4, 1, 3, 3, 2, 5], [0, 3, 4, 4, 5, 0], [2, 5, 5, 0, 3, 3], [1, 4, 0, 5, 0, 0]]

# Calculate mean rating per user.
means = np.zeros(6)

for i in range(6):
    non_empty = 0

    for rating in table[i]:
        if rating > 0:
            means[i] += rating
            non_empty += 1

    means[i] /= non_empty

# Calculate pairwise similarities between users using a modified Pearson correlation.
# Aggarwal, "Data Mining: The Textbook", equation 18.12
pearson = np.zeros((6, 6))

for i in range(6):
    for j in range(i + 1, 6):
        sum_x = 0
        sum_y = 0
        sum_xy = 0

        for k in range(6):
            if table[i][k] != 0 and table[j][k] != 0:
                sum_x += (table[i][k] - means[i])**2
                sum_y += (table[j][k] - means[j])**2
                sum_xy += (table[i][k] - means[i]) * (table[j][k] - means[j])

        pearson[i][j] = sum_xy / np.sqrt(sum_x) / np.sqrt(sum_y)
        pearson[j][i] = pearson[i][j]

# Get 2-nearest neighbors for each user.
nn = []
rs = []

for i in range(6):
    r1 = -1
    r2 = -1
    nn1 = -1
    nn2 = -1

    for j in range(6):
        if pearson[i][j] >= 0.5 and pearson[i][j] > r1:
            r2 = r1
            nn2 = nn1
            r1 = pearson[i][j]
            nn1 = j

        elif pearson[i][j] >= 0.5 and pearson[i][j] > r2:
            r2 = pearson[i][j]
            nn2 = j

    nn.append([nn1, nn2])
    rs.append([r1, r2])

normalized_table = [row[:] for row in table]

# Normalize table.
for i in range(6):
    for j in range(6):
        normalized_table[i][j] -= means[i]

# Predict missing ratings using 2-nearest neighbors.
for i in range(6):
    for j in range(6):
        if table[i][j] == 0 and nn[i][1] == -1:
            if table[nn[i][0]] != 0:
                table[i][j] = int(round(normalized_table[nn[i][0]][j] + means[i]))

        elif table[i][j] == 0:
            if table[nn[i][0]] == 0:
                if table[nn[i][1]] != 0:
                    table[i][j] = int(round(normalized_table[nn[i][1]][j] + means[i]))

            elif table[nn[i][0]] != 0:
                if table[nn[i][1]] == 0:
                    table[i][j] = int(round(normalized_table[nn[i][0]][j] + means[i]))

                else:
                    table[i][j] = int(round((rs[i][0] * normalized_table[nn[i][0]][j] + rs[i][1] *
                                             normalized_table[nn[i][1]][j]) / (rs[i][0] + rs[i][1]) + means[i]))

print(table)
