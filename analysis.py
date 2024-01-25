from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

MAX_K_VALUE = 10
iris_data = datasets.load_iris().data.tolist()


def main():
    inertia_values = []
    for k in range(1, 1 + MAX_K_VALUE):
        kmeans = KMeans(n_clusters=k, random_state=0, init="k-means++")
        kmeans.fit(iris_data)
        inertia_values.append(kmeans.inertia_)

    x_values = range(1, 1 + MAX_K_VALUE)
    y_values = inertia_values
    plt.plot(x_values, y_values)

    elbow_x = x_values[2]
    elbow_y = y_values[2]
    plt.plot(elbow_x, elbow_y, 'o', markersize=10, color='black', fillstyle='none')
    plt.annotate('Elbow \n Point', xy=(elbow_x + 0.1, elbow_y + 5), xytext=(elbow_x + 2, elbow_y + 200),
                 arrowprops=dict(facecolor='red', arrowstyle='->', linestyle='dashed', linewidth=1.5), color='black')

    plt.xlabel('k')
    plt.ylabel('Average Dispersion')
    plt.title('Elbow Method for selection of optimal "K" clusters')
    plt.grid(True)

    plt.savefig('elbow.png')


if __name__ == "__main__":
    main()
