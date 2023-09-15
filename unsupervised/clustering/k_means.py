# External libraries
import numpy as np


class KMeans:
    """An implementation of the K-Means clustering algorithm.

    Parameters:
        n_clusters: The number of clusters to form.
        max_iters: The maximum number of iterations to run K-Means.

    Attributes:
        n_clusters: The number of clusters to form.
        max_iters: The maximum number of iterations to run K-Means.
        centroids: The centroids of the clusters.

    Methods:
        fit(x): Fit K-Means clustering to the input data 'x'.
        transform(x): Assign data points to the nearest cluster centroids.
        fit_transform(x): Fit K-Means and return the cluster assignments for the
            input data 'x'.

    Private Methods:
        _assign_clusters(x): Assign data points to the nearest cluster centroid.
        _compute_centroids(x, labels): Compute new cluster centroids based on
            data points and labels.

    """

    def __init__(self, n_clusters: int = 8, max_iters: int = 300):
        """Initialize K-Means clustering with the specified number of clusters
            and maximum iterations.

        Args:
            n_clusters: The number of clusters to form.
            max_iters: The maximum number of iterations to run K-Means.

        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, x: np.array):
        """Fit K-Means clustering to the input data 'x'.

        Args:
            x: The input data.

        Returns:
            The fitted K-Means instance.

        """
        n_samples, n_features = x.shape
        np.random.seed(0)
        self.centroids = x[
            np.random.choice(n_samples, self.n_clusters, replace=False)
        ]

        for _ in range(self.max_iters):
            labels = self._assign_clusters(x)

            new_centroids = self._compute_centroids(x, labels)

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        return self

    def transform(self, x: np.array) -> np.array:
        """Assign data points in 'x' to the nearest cluster centroids.

        Args:
            x: The input data.

        Returns:
            labels: Cluster assignments for each data point.

        """
        labels = self._assign_clusters(x)
        return labels

    def fit_transform(self, x: np.array):
        """Fit K-Means to the input data 'x' and return cluster assignments.

        Args:
            x: The input data.

        Returns:
            labels: Cluster assignments for each data point.

        """
        self.fit(x)
        return self.transform(x)

    def _assign_clusters(self, x: np.array):
        """Assign data points in 'x' to the nearest cluster centroids.

        Args:
            x: The input data.

        Returns:
            labels: Cluster assignments for each data point.

        """
        distances = np.linalg.norm(x[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, x: np.array, labels: np.array):
        """Compute new cluster centroids based on data points and labels.

        Args:
            x: The input data.
            labels: Cluster assignments for each data point.

        Returns:
            new_centroids: Updated cluster centroids.

        """
        new_centroids = np.array(
            [x[labels == i].mean(axis=0) for i in range(self.n_clusters)]
        )
        return new_centroids
