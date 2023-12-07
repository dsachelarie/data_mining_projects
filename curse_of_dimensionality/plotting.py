import matplotlib.pyplot as plt
import numpy as np

"""
Plotting for curse_of_dimensionality.py
"""

def plot_contrasts(data: [], p_values: [], dimensionalities: []):
    for p in p_values:
        x = []
        y = []

        for k in dimensionalities:
            x.append(k)
            y.append(np.average(list(map(lambda sample: sample["contrast"], filter(lambda sample: sample["dimensionality"] == k and sample["p"] == p, data)))))

        plt.plot(x, y, label="p=" + str(p))

    plt.xlabel("k")
    plt.ylabel("relative contrast")
    plt.title("Relative contrast by dimensionality")
    plt.legend()
    plt.show()


def plot_averages(data: [], p_values: [], dimensionalities: []):
    for p in p_values:
        x = []
        y1 = []
        y2 = []
        y3 = []

        for k in dimensionalities:
            x.append(k)
            y1.append(np.average(list(map(lambda sample: sample["max_distance"], filter(lambda sample: sample["dimensionality"] == k and sample["p"] == p, data)))))
            y2.append(np.average(list(map(lambda sample: sample["min_distance"], filter(lambda sample: sample["dimensionality"] == k and sample["p"] == p, data)))))
            y3.append(np.average(list(map(lambda sample: sample["mean_distance"], filter(lambda sample: sample["dimensionality"] == k and sample["p"] == p, data)))))

        plt.plot(x, y1, label="max" + str(p))
        plt.plot(x, y2, label="min" + str(p))
        plt.plot(x, y3, label="average" + str(p))

        plt.xlabel("k")
        plt.ylabel("distance")
        plt.title("averages by dimensionality")
        plt.legend()
        plt.show()


def plot_variance(data: [], p_values: [], dimensionalities: []):
    for p in p_values:
        x = []
        y = []

        for k in dimensionalities:
            x.append(k)
            y.append(np.average(list(map(lambda sample: sample["var_distance"], filter(lambda sample: sample["dimensionality"] == k and sample["p"] == p, data)))))

        plt.plot(x, y)

        plt.xlabel("k")
        plt.ylabel("variance")
        plt.title("avg variance by dimensionality, p=" + str(p))
        plt.legend()
        plt.show()
