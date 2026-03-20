import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import sparse


def mixing_time(adj_matrix: np.array):
    """
    This function takes a weighted network adjacency matrix, and calculates the
    mixing time for the associated matrix.
    """
    try:
        adj_matrix = drop_empties(adj_matrix)
        transition_matrix = sparse.csr_matrix(
            np.array(
                [[element / sum(row) for element in row] for row in adj_matrix]
            )
        )

        # creates fake eigenvalues to allow looping
        eigVal = [1, 1]
        count = 0

        # loops to avoid erroneously returning a second eigenvalue of 1
        while round(eigVal[1].real, 10) == 1 or count < 10:

            # finds the two largest eigenvalues and associated eigenvectors
            eigVal, eigVec = sparse.linalg.eigs(
                transition_matrix, k=2, which="LR"
            )
            count += 1

        secondEig = eigVal[1].real
        mixing = float(1) / (1 - secondEig)

        return mixing

    except:
        return None


def drop_empties(adj_matrix: np.array):
    """
    Takes a network adjacency matrix, and drops any columns with all zero
    values (corresponding to a node with no connections)
    """
    try:
        row_drop = [row for row in adj_matrix if sum(row) > 0]
        col_drop = [row for row in np.transpose(row_drop) if sum(row) > 0]
        return np.transpose(col_drop)
    except:
        return None


if __name__ == "__main__":
    fm_times = []
    icl_times = []
    for entry in os.scandir("graph_data"):
        adj_matrix = np.genfromtxt(entry.path, delimiter=",")
        mix_time = mixing_time(adj_matrix)
        if entry.path.startswith("graph_data/fm") and mix_time is not None:
            fm_times.append(np.log(mix_time))
        elif entry.path.startswith("graph_data/icl") and mix_time is not None:
            icl_times.append(np.log(mix_time))
    sns.kdeplot(x=fm_times, label="fm graphs")
    sns.kdeplot(x=icl_times, label="icl_graphs")
    plt.xlabel("Log Mixing Time")
    plt.legend()
    plt.savefig("mixing_times.png")
