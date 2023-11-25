import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances

from sklearn.datasets import make_classification

class CopKMeans(BaseEstimator, ClassifierMixin):
    def __init__(self, n_clusters, max_iter=1000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def violate_constraints(self, data_point, cluster_i, assigned_centroids,
                            must_link, cannot_link):
        for pt1, pt2 in must_link:
            if not pt1 == data_point and not pt2 == data_point:
                continue
            if pt2 == data_point:
                pt1, pt2 = pt2, pt1
            if assigned_centroids[pt2] == -1 :
                # we only enforce ML constraints if the other point has been assigned
                continue
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
            # FOR TESTING PURPOSE, TO BE DELETED
            if i%50 == 0: 
                print(f"Itération {i}")
            
            assigned_centroids = [-1]*len(X)
            for j, data_point in enumerate(X):
                # Calculate the closest clusters
                data_point = data_point.reshape(1, -1)
                # data_point is (1, n_features)
                # centroids is (n_clusters, n_features)
                distances = euclidean_distances(centroids, data_point.reshape(1,-1))
                # distances is (n_clusters, 1)
                closest_centroids = np.argsort(distances, axis=0).reshape(-1)

                # Check the constraints
                # assigned_centroid = -1
                for c in closest_centroids:
                    if not self.violate_constraints(data_point, clusters[c], assigned_centroids, must_link, cannot_link):
                        assigned_centroids[j] = c
                        break
                if assigned_centroids[j] == -1:
                    # Algorithm failed
                    return None

                # Assign to the first cluster that doesn't fail the constraints
                clusters[assigned_centroids[j]].append(data_point)

            # Update centroids
            centroids = np.array([np.mean(x, axis=0) for x in clusters]).reshape(centroids.shape)

        # Store the info
        self.centroids = centroids
        
        # Return the clusters
        return self

    def predict(self, X):
        # self_centroids is (n_clusters, n_features)
        # X is (n_points, n_features)
        # distance must be (n_points, n_clusters)
        distances = euclidean_distances(self.centroids, X)
        closest_clusters = np.argmin(distances, axis=0).reshape(-1)
        return closest_clusters

if __name__ == '__main__':
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=2111)
    model = CopKMeans(3)
    model.fit(X, [], [])