import numpy as np

"""
An exploration of how can numerical and categorical distance measures be combined.
The distance measures used were: L2 norm after performing min-max scaling for numerical values,
Goodall distance for categorical values.
"""

# Data about 6 cows
age = [2, 2, 5, 4, 7, 8]
milk = [10, 15, 20, 25, 35, 45]
race = ["H", "A", "H", "A", "F", "A"]
character = ["c", "l", "c", "k", "c", "k"]
music = ["ro", "co", "cl", "ro", "cl", "co"]

# Min-max scaling - Aggarwal, "Data Mining: The Textbook", 2.3.3
min_max_age = []
min_max_milk = []

for sample in age:
    min_max_age.append((sample - min(age)) / (max(age) - min(age)))

for sample in milk:
    min_max_milk.append((sample - min(milk)) / (max(milk) - min(milk)))

print(min_max_age)
print(min_max_milk)

# Pairwise Euclidian distances between cows
distances = np.zeros((6, 6))

for i in range(6):
    for j in range(i + 1, 6):
        distances[i][j] = np.sqrt((min_max_age[i] - min_max_age[j])**2 + (min_max_milk[i] - min_max_milk[j])**2)

print(distances)

# Goodall distance (1 - Goodall similarity) - Aggarwal, "Data Mining: The Textbook", 3.2.2
goodall = np.zeros((6, 6))

for i in range(6):
    for j in range(i + 1, 6):
        if race[i] == race[j]:
            p = 0
            for k in range(6):
                if race[i] == race[k]:
                    p += 1

            p /= 6
            goodall[i][j] += (1 - p**2)

        if character[i] == character[j]:
            p = 0
            for k in range(6):
                if character[i] == character[k]:
                    p += 1

            p /= 6
            goodall[i][j] += (1 - p**2)

        if music[i] == music[j]:
            p = 0
            for k in range(6):
                if music[i] == music[k]:
                    p += 1

            p /= 6
            goodall[i][j] += (1 - p**2)

        goodall[i][j] /= 3
        goodall[i][j] = 1 - goodall[i][j]

print(goodall)

# Combine Euclidean and Goodall distances - Aggarwal, "Data Mining: The Textbook", 3.2.3
combined = np.zeros((6, 6))
lmbda = 2/5

flattened_distances = []
flattened_goodall = []

for i in range(6):
    for j in range(i + 1, 6):
        flattened_distances.append(distances[i][j])
        flattened_goodall.append(goodall[i][j])

for i in range(6):
    for j in range(i + 1, 6):
        combined[i][j] = lmbda * distances[i][j] / np.std(flattened_distances) + (1 - lmbda) * goodall[i][j] / np.std(flattened_goodall)

print(combined)
