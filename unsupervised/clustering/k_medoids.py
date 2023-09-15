# External libraries
import numpy as np


class KMedoids:
    """An implementation of the K-Medoids clustering algorithm.

    Parameters:
        n_clusters: The number of clusters to form.
        max_iters: The maximum number of iterations to run K-Medoids.

    Attributes:
        n_clusters: The number of clusters to form.
        max_iters: The maximum number of iterations to run K-Medoids.
        medoids: The medoids of the clusters.

    Methods:
        fit(X): Fit K-Medoids clustering to the input data 'X'.
        transform(X): Assign data points to the nearest medoids.
        fit_transform(X): Fit K-Medoids and return the medoid assignments for
            the input data 'X'.

    Private Methods:
        _assign_medoids(X): Assign data points to the nearest medoids.
        _compute_medoids(X, labels): Compute new medoids based on data points
            and labels.

    """

    def __init__(self, n_clusters: int = 8, max_iters: int = 1000):
        """Initialize K-Medoids clustering with the specified number of clusters
            and maximum iterations.

        Args:
            n_clusters: The number of clusters to form.
            max_iters: The maximum number of iterations to run K-Medoids.

        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.medoids = None

    def fit(self, x: np.array):
        """Fit K-Medoids clustering to the input data 'X'.

        Args:
            x: The input data.

        Returns:
            self: The fitted K-Medoids instance.

        """
        n_samples, _ = x.shape
        np.random.seed(0)
        self.medoids = x[
            np.random.choice(n_samples, self.n_clusters, replace=False)
        ]

        for _ in range(self.max_iters):
            # Assign each point to the nearest medoid
            labels = self._assign_medoids(x)

            # Compute new medoids
            new_medoids = self._compute_medoids(x, labels)

            # Check for convergence
            if np.all(self.medoids == new_medoids):
                break

            self.medoids = new_medoids

        return self

    def transform(self, x: np.array) -> np.array:
        """Assign data points in 'X' to the nearest medoids.

        Args:
            x: The input data.

        Returns:
            labels: Medoid assignments for each data point.

        """
        labels = self._assign_medoids(x)
        return labels

    def fit_transform(self, x: np.array) -> np.array:
        """Fit K-Medoids to the input data 'X' and return medoid assignments.

        Args:
            x: The input data.

        Returns:
            labels (ndarray): Medoid assignments for each data point.

        """
        self.fit(x)
        return self.transform(x)

    def _assign_medoids(self, x: np.array) -> np.array:
        """Assign data points in 'X' to the nearest medoids.

        Args:
            x: The input data.

        Returns:
            labels: Medoid assignments for each data point.

        """
        distances = np.linalg.norm(x[:, np.newaxis] - self.medoids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_medoids(self, x: np.array, labels: np.array) -> np.array:
        """Compute new medoids based on data points and labels.

        Args:
            x: The input data.
            labels: Medoid assignments for each data point.

        Returns:
            new_medoids (ndarray): Updated medoids.

        """
        new_medoids = []

        # Iterate over each cluster (i)
        for i in range(self.n_clusters):
            cluster_data = x[labels == i]  # Get data points in cluster i

            # Calculate pairwise absolute differences between data points i
            # n the cluster
            pairwise_differences = np.abs(
                cluster_data - cluster_data[:, np.newaxis]
            )

            # Sum the absolute differences along axis 2 to get distances
            distances = np.sum(pairwise_differences, axis=2)

            # Find the index of the data point with the minimum distance
            # (the new medoid)
            new_medoid_index = np.argmin(distances)

            # Get the new medoid from the cluster
            new_medoid = cluster_data[new_medoid_index]

            # Append the new medoid to the list
            new_medoids.append(new_medoid)

        # Convert the list of new medoids to a NumPy array
        new_medoids = np.array(new_medoids)

        return new_medoids