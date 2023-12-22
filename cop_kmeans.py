import copy
import networkx as nx
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.datasets import make_classification

class CopKMeans(BaseEstimator, ClassifierMixin):
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def violate_constraints(self, idx_data_point, c, assigned_centroids, cannot_link):
        for neighbour in list(self.G.adj[idx_data_point]):
            if assigned_centroids[neighbour] == -1 :
                # we only enforce ML constraints if the other point has been assigned
                continue
            if assigned_centroids[neighbour] != c:
                # print(idx_data_point, neighbour, "ml")
                return True

        for pt in cannot_link[idx_data_point]:
            if assigned_centroids[pt] == c :
                # print(idx_data_point, pt, "cl")
                return True

        return False
    
    def transitive_closure(self, n, must_link, cannot_link):
        # Closure of must_link
        G = nx.Graph()
        G.add_nodes_from(list(range(n)))
        G.add_edges_from(must_link)
        G = nx.transitive_closure(G)
        
        # Closure of cannot_link
        cl = [set() for _ in range(n)]
        for i, tpl in enumerate(cannot_link):
            pt1, pt2 = tpl
            cl[pt1].add(pt2)
            cl[pt2].add(pt1)
            
            # pt1 neighbours cannot link with pt2
            for pt1_n in list(G.adj[pt1]):
                cl[pt2].add(pt1_n)
                cl[pt1_n].add(pt2)
            
            # pt2 neighbours cannot link with pt1
            for pt2_n in list(G.adj[pt2]):
                cl[pt1].add(pt2_n)
                cl[pt2_n].add(pt1)
                
            # pt1 neighbours cannot link with pt2 neighbours
            for pt1_n in list(G.adj[pt1]):
                for pt2_n in list(G.adj[pt2]):
                    cl[pt2_n].add(pt1_n)
                    cl[pt1_n].add(pt2_n)
        
        
        self.G = G
        return cl

    def fit(self, X, must_link, cannot_link):
        # Transitive closure over the constraints
        cannot_link = self.transitive_closure(X.shape[0], must_link, cannot_link)
           
        # Initialization
        centroids_idx = np.random.choice(list(range(X.shape[0])),
                                         self.n_clusters, replace=False)
        centroids = X[centroids_idx]
        clusters = [[] for _ in range(self.n_clusters)]

        for i in range(self.max_iter):            
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
                    if not self.violate_constraints(j, c, assigned_centroids, cannot_link):
                        assigned_centroids[j] = c
                        break
                if assigned_centroids[j] == -1:
                    # Algorithm failed
                    # print(j, data_point)
                    # print("FAIL")
                    return None

                # Assign to the first cluster that doesn't fail the constraints
                clusters[assigned_centroids[j]].append(data_point)

            # Update centroids
            prev_centroids = copy.deepcopy(centroids)
            centroids = np.array([np.mean(x, axis=0) for x in clusters]).reshape(centroids.shape)

            # Check for convergence
            # Same method used in the already implemented method for comparison purpose
            converged = np.allclose((prev_centroids-centroids), np.zeros(centroids.shape), atol=1e-6, rtol=0)
            if converged :
                print(i, "iterations")
                break

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
    # FOR TESTING PURPOSES, TO BE DELETED
    ml = []
    cl = []
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=2111)

    for i, pt1 in enumerate(y):
        for j, pt2 in enumerate(y[i+1:]):
            if pt1 == pt2 :
                ml.append((i, j+i+1))
            else:
                cl.append((i, j+i+1))

    print(ml)
    print(cl)
    print(len(ml), len(cl))

    ml, cl = np.array(ml), np.array(cl)
    ml_subset = ml[np.random.choice(len(ml), int(0.1*len(ml)))]
    cl_subset = cl[np.random.choice(len(cl), int(0.1*len(cl)))]

    # ml_subset = ml[np.random.choice(len(ml), 10)]
    # cl_subset = cl[np.random.choice(len(cl), 10)]
    model = CopKMeans(3, 200)

    model.fit(X, ml_subset, cl_subset)
    
    print(1)