import pandas as pd
import numpy as np
import mykmeanssp
import sys

# mykmeanssp.fit(PyD_vectors, PyCentroids, max_iter, eps)

DEFAULT_ITER = 300
ITER_MAX_VALUE = 999
ITER_MIN_VALUE = 2
K_MIN_VALUE = 2
ERR_MSG_K = "Invalid number of clusters!"
ERR_MSG_ITER = "Invalid maximum max_iteration!"
ERR_MSG_DEFAULT = "An Error Has Occurred"


def main(argv):
    k, max_iter, eps, d_vectors = collect_argv(argv)

    check_arguments(k, max_iter, len(d_vectors))
    centroids, centroids_indexes = init_centroids(d_vectors, k)

    final_centroids = mykmeanssp.fit(d_vectors.tolist(), centroids.tolist(), max_iter, eps)
    if final_centroids is None:
        print(ERR_MSG_DEFAULT)
        exit()

    print_output(final_centroids, centroids_indexes)


def collect_argv(argv):
    # NOTE: argv = {kmeans_pp, k, max_iter (optional), eps, file_name_1, file_name_2}

    if not (5 <= len(argv) <= 6):
        print(ERR_MSG_DEFAULT)
        exit()

    k = int(argv[1])
    max_iter = int(argv[2]) if (len(argv) == 6) else DEFAULT_ITER
    eps = float(argv[-3])
    file_name_1 = argv[-2]
    file_name_2 = argv[-1]
    d_vectors = read_csv_files(file_name_1, file_name_2)

    return k, max_iter, eps, d_vectors


def print_output(final_centroids, centroids_indexes):
    indexes_str = ','.join(str(i) for i in centroids_indexes)
    print(indexes_str)

    for centroid in final_centroids:
        centroid_values = [f'{float(value):.4f}' for value in centroid]
        centroid_str = ','.join(centroid_values)
        print(centroid_str)


def init_centroids(d_vectors, k):
    np.random.seed(0)
    centroids = []
    centroids_indexes = []

    d_vectors_indexes = [i for i in range(len(d_vectors))]

    choice_index = np.random.choice(d_vectors_indexes)
    choice = d_vectors[choice_index].tolist()

    centroids_indexes.append(choice_index)
    centroids.append(choice)

    # NOTE: d_vector that has been chosen to be a centroid
    # will carry probability to be chosen again = 0
    # since it's closest d_vector is itself, and removing it is not necessary

    for i in range(k - 1):
        distances = compute_distances(centroids, d_vectors)
        WPD = distances / np.sum(distances)  # Weighted Probability Distribution

        choice_index = np.random.choice(len(d_vectors), p=WPD)
        choice = d_vectors[choice_index].tolist()

        centroids_indexes.append(choice_index)
        centroids.append(choice)

    return np.array(centroids), centroids_indexes


def compute_distances(centroids, d_vectors):
    return [min(np.linalg.norm(centroid - d_vector) for centroid in centroids) for d_vector in d_vectors]


def check_arguments(k, max_iter, number_of_d_vectors):
    k_invalid = not (K_MIN_VALUE <= k < number_of_d_vectors)
    max_iter_invalid = not (ITER_MIN_VALUE <= max_iter <= ITER_MAX_VALUE)
    if k_invalid:
        print(ERR_MSG_K)
    if max_iter_invalid:
        print(ERR_MSG_ITER)

    if k_invalid or max_iter_invalid:
        exit()


def read_csv_files(file_name_1, file_name_2):
    try:
        data1 = pd.read_csv(file_name_1, header=None, index_col=0)
        data2 = pd.read_csv(file_name_2, header=None, index_col=0)

    except:
        print(ERR_MSG_DEFAULT)
        exit()

    return np.array(pd.merge(data1, data2, sort=True, left_index=True, right_index=True))


if __name__ == "__main__":
    main(sys.argv)
