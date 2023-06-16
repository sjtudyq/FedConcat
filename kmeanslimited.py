import numpy as np
from sklearn.cluster import KMeans

def constrained_kmeans(X, K, max_cluster_size, num_iters=100):
    N, D = X.shape

    # Start with a random initialization of cluster centroids using KMeans.
    centroids = KMeans(n_clusters=K).fit(X).cluster_centers_

    for _ in range(num_iters):
        # Compute distances from points to centroids.
        distances = np.linalg.norm(X[:, None] - centroids, axis=-1)

        # Assign points to closest centroid.
        cluster_assignment = np.argmin(distances, axis=1)

        # If some clusters are empty, reinitialize their centroids.
        for j in range(K):
            if (cluster_assignment == j).sum() == 0:
                centroids[j] = X[np.random.choice(N)]

        # If a cluster is over capacity, reassign its furthest points.
        for j in range(K):
            while (cluster_assignment == j).sum() > max_cluster_size:
                idx = np.where(cluster_assignment == j)[0]
                furthest_point_idx = idx[np.argmax(distances[idx, j])]
                cluster_assignment[furthest_point_idx] = -1
                distances[furthest_point_idx, j] = np.inf

        # If a point is unassigned, assign it to the closest under-capacity cluster.
        for i in range(N):
            if cluster_assignment[i] == -1:
                for j in np.argsort(distances[i]):
                    if (cluster_assignment == j).sum() < max_cluster_size:
                        cluster_assignment[i] = j
                        break

        # Update centroids.
        for j in range(K):
            if (cluster_assignment == j).sum() > 0:
                centroids[j] = X[cluster_assignment == j].mean(axis=0)

    return cluster_assignment
