import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances


class CopKMeans(BaseEstimator, ClassifierMixin):
    def __init__(self, n_clusters, max_iter=1000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def violate_constraints(self, data_point, cluster_i,
                            must_link, cannot_link):
        # TODO : probably doesn't work because no points are assigned
        # to clusters at initialization so must_link constraints
        # will always be false. There will be a need to add an order
        for pt1, pt2 in must_link:
            if not pt1 == data_point and not pt2 == data_point:
                continue
            if pt2 == data_point:
                pt1, pt2 = pt2, pt1
            if pt2 not in cluster_i:
                return True

        for pt1, pt2 in cannot_link:
            if not pt1 == data_point and not pt2 == data_point:
                continue
            if pt2 == data_point:
                pt1, pt2 = pt2, pt1
            if pt2 in cluster_i:
                return True

        return False

    def fit(self, X, must_link, cannot_link):
        # Initialization
        centroids_idx = np.random.choice(list(range(X.shape[0])),
                                         self.n_clusters, replace=False)
        centroids = X[centroids_idx]
        clusters = [[] for _ in range(self.n_clusters)]
        # TODO : CHECK WHEN TO STOP ? WHEN IS THERE CONVERGENCE
        for i in range(self.max_iter):
            for j, data_point in enumerate(X):
                # Calculate the closest clusters
                distances = euclidean_distances(centroids, data_point)
                closest_centroids = np.argsort(distances, axis=0)

                # Check the constraints
                assigned_centroid = -1
                for c in closest_centroids:
                    if not self.violate_constraints(data_point, clusters[c],
                                                    must_link, cannot_link):
                        assigned_centroid = c
                        break
                if assigned_centroid == -1:
                    # Algorithm failed
                    return None

                # Assign to the first cluster that doesn't fail the constraints
                clusters[j].append(data_point)

            # Update centroids
            centroids = [np.mean(x, axis=0) for x in clusters]

        # Return the clusters
        return clusters

    def predict(self, X):
        pass
