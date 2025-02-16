import math
from sys import argv, stderr

import numpy as np


def generate_correct_matrix(size):
    rng = np.random.default_rng()
    mat = np.tril(rng.normal(0, 10, size=(size, size)))
    for i in range(size):
        for j in range(i + 1, size):
            mat[i, j] = 0
    result = mat @ mat.T
    return result


def read_lower_triangular_matrix(filename):
    try:
        with open(filename, "r") as f:
            size = int(f.readline())
            mat = np.zeros((size, size))
            values = f.readline().split(" ")
            values = [float(i) for i in values]
            for i in range(size):
                for j in range(size):
                    mat[i, j] = values[i * size + j]
        return mat
    except IOError:
        print("Error reading file", file=stderr)
        exit(-1)


def diagonal(mat, l_mat, i, j):
    negative = 0
    for k in range(j):
        result = math.pow(get_value(mat, l_mat, j, k), 2)
        negative += result
    value = math.sqrt(mat[i, j] - negative)
    return value


def not_diagonal(mat, l_mat, i, j):
    denom = get_value(mat, l_mat, j, j)
    negative = 0
    for k in range(j):
        val = get_value(mat, l_mat, i, k)
        val *= get_value(mat, l_mat, j, k)
        negative += val
    return (1.0 / denom) * (mat[i, j] - negative)


def get_value(mat, l_mat, i, j):
    if l_mat[i, j] < np.inf:
        return l_mat[i, j]

    if i == j:
        return diagonal(mat, l_mat, i, j)
    else:
        return not_diagonal(mat, l_mat, i, j)


def cholesky(mat):
    size = len(mat)
    l_mat = np.full(mat.shape, np.inf)
    for i in range(size):
        for j in range(size):
            if j > i:
                l_mat[i, j] = 0
            else:
                value = get_value(mat, l_mat, i, j)
                l_mat[i, j] = value
    return l_mat


def print_matrix(mat):
    size = len(mat)
    for i in range(size):
        for j in range(size):
            print(f"{mat[i][j]:3}", end=" ")
        print("")


def main(args):
    size = int(argv[1])
    mat = generate_correct_matrix(size)
    print(mat)
    print("***")
    l_mat = cholesky(mat)
    print(l_mat)
    print("***")
    comp = np.linalg.cholesky(mat)
    print(comp)
    print(np.allclose(comp, l_mat))


if __name__ == "__main__":
    main(argv)
