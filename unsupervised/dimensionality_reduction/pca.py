import numpy as np


class PCA:
    """Principal Component Analysis (PCA) for dimensionality reduction.

    This class implements PCA using NumPy to perform dimensionality reduction on
    a given dataset.

    Args:
        n_components: The number of principal components to retain.

    Attributes:
        n_components: The number of principal components to retain.
        components: The retained principal components.
        mean: The mean vector of the input data.
        std: The standard deviation vector of the input data.

    Methods:
        fit(matrix): Fit the PCA model to the input data.
        transform(matrix): Project the input data onto the principal components.
        fit_transform(matrix): Fit the PCA model to the input data and
            transform the data in the PCA space.

    Example:
        # Create a PCA instance with 2 principal components
        pca = PCA(n_components=2)

        # Fit the PCA model to the data
        pca.fit(data_matrix)

        # Transform the data into the PCA space
        transformed_data = pca.transform(data_matrix)

    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, matrix: np.array) -> None:
        """Fit the PCA model to the input data.

        Args:
            matrix: The input data matrix where each row
                represents a data point and each column represents a feature.
        """

        self.mean = np.mean(matrix, axis=0)

        matrix = (matrix - self.mean)

        # Compute the covariance matrix
        cov_matrix = np.cov(matrix, rowvar=False)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort the eigenvalues and eigenvectors in decreasing order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store the first n_components eigenvectors as the principal components
        self.components = eigenvectors[:, : self.n_components]

    def fit_transform(self, matrix):
        """Fit the PCA model to the input data and transform the data.

        Args:
            matrix: The input data matrix where each row represents a data
             point and each column represents a feature.

        Returns:
            The transformed data in the PCA space.

        """
        # Fit the PCA model to the data
        self.fit(matrix)

        # Transform the data in the PCA space
        return self.transform(matrix)

    def transform(self, matrix: np.array) -> np.array:
        """Project the input data onto the principal components.

        Args:
            matrix: The input data matrix where each row
                represents a data point and each column represents a feature.

        Returns:
            The transformed data in the PCA space.

        """
        matrix = (matrix - self.mean)

        # Project the data onto the principal components
        return np.dot(matrix, self.components)