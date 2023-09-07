import numpy as np

from typing import Union
from typing import Tuple


class TSNE:
    """t-Distributed Stochastic Neighbor Embedding (t-SNE) for dimensionality
        reduction.

    This class implements t-SNE, a technique for reducing the dimensionality of
    data while preserving pairwise similarities. It is commonly used for
    visualization of     high-dimensional data in a lower-dimensional space.

    Args:
        n_components: Number of components (dimensions) in the embedded space.
        target_perplexity: The target perplexity of the embedded space.
            Perplexity is a hyperparameter that balances preserving global and
            local structures.
        learning_rate : The learning rate for gradient descent optimization.
        n_iter: The number of iterations for the optimization process.
        seed: Random seed for reproducibility.
        momentum: The momentum parameter for gradient descent.

    Example:
        # Create a TSNE instance with 2 components and a target perplexity of 30
        tsne = TSNE(n_components=2, target_perplexity=30)

        # Fit and transform the high-dimensional data matrix X
        X_embedded = tsne.fit_transform(X)

    """

    def __init__(
        self,
        n_components: int = 2,
        target_perplexity: float = 12.0,
        learning_rate: int = 12,
        n_iter: int = 1000,
        seed: int = 42,
        momentum: float = 0.9,
    ):
        self.n_components = n_components
        self.target_perplexity = target_perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.seed = seed
        self.tse = True
        self.momentum = momentum

    @staticmethod
    def neg_squared_euc_dists(matrix: np.array) -> np.array:
        """Compute matrix containing negative squared euclidean
            distance for all pairs of points in input matrix X.

        Arguments:
            matrix: matrix of size NxD.

        Returns:
            NxN matrix D, with entry D_ij = negative squared
            euclidean distance between rows X_i and X_j.

        """
        sum_matrix = np.sum(np.square(matrix), 1)
        d = np.add(
            np.add(-2 * np.dot(matrix, matrix.T), sum_matrix).T, sum_matrix
        )
        return -d

    @staticmethod
    def softmax(matrix: np.array, diagonal_zero: bool = True) -> np.array:
        """Apply the softmax function to a matrix.

        The softmax function is used to convert the input matrix into a
        probability distribution over its rows (each row represents a data
        point). It exponentiates the input values and normalizes them to
        produce a probability distribution.

        Args:
            matrix: The input matrix to which the softmax function is applied.
            diagonal_zero: Whether to set diagonal elements to zero before
                applying the softmax.
                This is useful when applying softmax to similarity matrices.

        Returns:
            A matrix with the same shape as the input, where each row is a
            probability distribution obtained by applying softmax to the
            corresponding row of the input matrix.

        """
        e_x = np.exp(matrix - np.max(matrix, axis=1).reshape([-1, 1]))
        if diagonal_zero:
            np.fill_diagonal(e_x, 0.0)
        e_x = e_x + 1e-8
        return e_x / e_x.sum(axis=1).reshape([-1, 1])

    @staticmethod
    def calc_prob_matrix(
        distances: np.array, sigmas: Union[np.array, None] = None
    ) -> np.array:
        """Calculate the probability matrix based on pairwise distances.

        This method calculates a probability matrix that represents the
        conditional probabilities of selecting a data point as a neighbor for
        each data point based on pairwise distances.

        Args:
            distances: Pairwise distances between data points.
            sigmas: An array of bandwidths (sigma) for the Gaussian kernel.
                If provided, these bandwidths are used in the calculation.
                If None, the default softmax function is used for probability
                calculation.

        Returns:
            A probability matrix where each element (i, j) represents the
            conditional probability of selecting data point j as a neighbor of
             data point i.

        """
        if sigmas is not None:
            two_sig_sq = 2.0 * np.square(sigmas.reshape((-1, 1)))
            return TSNE.softmax(distances / two_sig_sq)
        else:
            return TSNE.softmax(distances)

    @staticmethod
    def binary_search(
        eval_fn: callable,
        target: float,
        tol: float = 1e-10,
        max_iter: int = 10000,
        lower: float = 1e-20,
        upper: float = 1000.0,
    ) -> float:
        """Perform a binary search to find a target value.

        This method performs a binary search to find a value that satisfies a
        specified target value within a given tolerance.

        Args:
            eval_fn: A function that takes a single argument (the guess value)
                and returns a value to be compared with the target.
            target: The target value to be reached.
            tol: The tolerance level for the difference between the guessed
                value and the target value.
                The search stops when the absolute difference is
                less than or equal to this tolerance.
            max_iter: The maximum number of iterations for the binary search.
            lower: The lower bound for the search range.
            upper: The upper bound for the search range.

        Returns:
            The estimated value that satisfies the target within the specified
            tolerance, or the best estimate found after the maximum number of
            iterations.

        """

        for i in range(max_iter):
            guess = (lower + upper) / 2.0
            val = eval_fn(guess)
            if val > target:
                upper = guess
            else:
                lower = guess
            if np.abs(val - target) <= tol:
                break
        return guess

    @staticmethod
    def calc_perplexity(prob_matrix: np.array) -> np.array:
        """Calculate the perplexity of a probability matrix.

        Perplexity is a measure of how well a probability distribution predicts
        a dataset. It quantifies how surprised one would be by a random event
        drawn from the distribution.
        In the context of t-SNE, perplexity is a hyperparameter used to
        balance preserving global and local structures.

        Args:
            prob_matrix: A probability matrix representing conditional
                probabilities between data points.

        Returns:
            The perplexity score calculated from the input probability matrix.

        """
        entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
        perplexity = 2**entropy
        return perplexity

    @staticmethod
    def perplexity(distances: np.array, sigmas: Union[np.array, None]) -> float:
        """Calculate perplexity using pairwise distances and sigmas.

        This static method calculates the perplexity value using pairwise
        distances and sigmas as input.
        Perplexity is a measure of how well a probability distribution
        predicts a dataset.
        It quantifies how surprised one would be by a random event
        drawn from the distribution.

        Args:
            distances: Pairwise distances between data points.
            sigmas: An array of bandwidths (sigma) for the Gaussian
                kernel. If provided, these bandwidths are used in the
                calculation. If None, the default softmax function is used for
                probability calculation.

        Returns:
            float: The perplexity score calculated based on the input distances
                and sigmas.

        """
        return TSNE.calc_perplexity(TSNE.calc_prob_matrix(distances, sigmas))

    @staticmethod
    def find_optimal_sigmas(
        distances: np.array, target_perplexity: float
    ) -> np.array:
        """Find optimal sigmas for the given pairwise distances and target
            perplexity.

        This method iterates through each data point and finds an optimal sigma
        value that achieves the target perplexity when used in the perplexity
        calculation. It uses binary search to find the optimal sigma.

        Args:
            distances: Pairwise distances between data points.
            target_perplexity: The target perplexity value to be achieved.

        Returns:
            An array of optimal sigma values corresponding to each data point.

        """
        sigmas = []
        for i in range(distances.shape[0]):
            eval_func = lambda sigma: TSNE.perplexity(
                distances[i : i + 1, :], np.array(sigma)
            )
            correct_sigma = TSNE.binary_search(eval_func, target_perplexity)
            sigmas.append(correct_sigma)
        return np.array(sigmas)

    @staticmethod
    def p_conditional_to_joint(array: np.array) -> np.array:
        """Convert conditional probabilities to joint probabilities.

        This static method takes a matrix of conditional probabilities (P) and
        converts them into joint probabilities. It calculates the joint
        probabilities by averaging
        the conditional probabilities and symmetrizing the result.

        Args:
            array: A matrix of conditional probabilities.

        Returns:
            A matrix of joint probabilities.

        """
        return (array + array.T) / (2.0 * array.shape[0])

    @staticmethod
    def p_joint(matrix: np.array, target_perplexity: float) -> np.array:
        """Calculate the joint probability distribution for t-SNE.

        This method calculates the joint probability distribution for t-SNE
        using the input data matrix and a target perplexity value.
        It involves several steps:
        1. Computes pairwise Euclidean distances between data points.
        2. Finds optimal sigma values for each data point to achieve the target
            perplexity.
        3. Computes conditional probabilities based on pairwise distances and
            sigmas.
        4. Converts conditional probabilities to joint probabilities.

        Args:
            matrix: The input data matrix.
            target_perplexity: The target perplexity value to be achieved.

        Returns:
            The joint probability distribution matrix.

        """
        distances = TSNE.neg_squared_euc_dists(matrix)
        sigmas = TSNE.find_optimal_sigmas(distances, target_perplexity)
        p_conditional = TSNE.calc_prob_matrix(distances, sigmas)
        p = TSNE.p_conditional_to_joint(p_conditional)
        return p

    @staticmethod
    def q_joint(matrix_y: np.array) -> Tuple[np.array, None]:
        """Calculate the joint probability distribution for the low-dimensional space.

        This method calculates the joint probability distribution for the low-dimensional
        space in t-SNE using the input data matrix Y. It involves several steps:
        1. Computes pairwise squared Euclidean distances between data points in the
           low-dimensional space.
        2. Exponentiates the negative squared distances.
        3. Sets the diagonal elements of the exponentiated distances to zero.
        4. Normalizes the exponentiated distances to obtain the joint probability
           distribution.

        Args:
            matrix_y (np.array): The low-dimensional data matrix.

        Returns:
            A tuple containing the joint probability distribution
            matrix and a None value (for compatibility with other methods).

        """
        distances = TSNE.neg_squared_euc_dists(matrix_y)
        exp_distances = np.exp(distances)
        np.fill_diagonal(exp_distances, 0.0)
        return exp_distances / np.sum(exp_distances), None

    @staticmethod
    def symmetric_sne_grad(
        p: np.array, q: np.array, y: np.array, _
    ) -> np.array:
        """Calculate the gradient of symmetric SNE loss.

        This static method calculates the gradient of the symmetric SNE
        (Symmetric Stochastic Neighbor Embedding) loss with respect to the
        low-dimensional data points (Y).
        It is used in the optimization process of t-SNE to find the optimal
        low-dimensional representation of data points.

        Args:
            p: The joint probability distribution matrix.
            q: The low-dimensional probability distribution matrix.
            y: The low-dimensional data matrix.
            _ (optional): Ignored parameter (for compatibility).

        Returns:
            np.array: The gradient of the symmetric SNE loss with respect to the
            low-dimensional data points (Y).

        """
        pq_diff = p - q
        pq_expanded = np.expand_dims(pq_diff, 2)
        y_diffs = np.expand_dims(y, 1) - np.expand_dims(y, 0)
        grad = 4.0 * (pq_expanded * y_diffs).sum(1)
        return grad

    @staticmethod
    def q_tsne(matrix_y: np.array) -> Tuple[np.array, np.array]:
        """Calculate the joint probability distribution for t-SNE.

        This static method calculates the joint probability distribution for
        t-SNE using the input data matrix Y. It involves several steps:
        1. Computes pairwise squared Euclidean distances between data points in
            the low-dimensional space.
        2. Calculates the inverse of (1 - distances) to obtain the joint
            probability distribution.
        3. Sets the diagonal elements of the joint probability distribution to
            zero.
        4. Normalizes the joint probability distribution.

        Args:
            matrix_y: The low-dimensional data matrix.

        Returns:
            A tuple containing the joint probability distribution
            matrix and the inverse of distances (for compatibility).

        """
        distances = TSNE.neg_squared_euc_dists(matrix_y)
        inv_distances = np.power(1.0 - distances, -1)
        np.fill_diagonal(inv_distances, 0.0)
        return inv_distances / np.sum(inv_distances), inv_distances

    @staticmethod
    def tsne_grad(
        p: np.array, q: np.array, y: np.array, inv_distances: np.array
    ) -> np.array:
        """Calculate the gradient of t-SNE loss.

        This static method calculates the gradient of the t-SNE loss with
        respect to the low-dimensional data points (Y).
        It is used in the optimization process of t-SNE
        to find the optimal low-dimensional representation of data points.

        Args:
            p: The joint probability distribution matrix.
            q: The low-dimensional probability distribution matrix.
            y: The low-dimensional data matrix.
            inv_distances: The inverse of distances (for compatibility).

        Returns:
            The gradient of the t-SNE loss with respect to the low-dimensional
            data points (Y).

        """
        pq_diff = p - q
        pq_expanded = np.expand_dims(pq_diff, 2)
        y_diffs = np.expand_dims(y, 1) - np.expand_dims(y, 0)
        distances_expanded = np.expand_dims(inv_distances, 2)
        y_diffs_wt = y_diffs * distances_expanded
        grad = 4.0 * (pq_expanded * y_diffs_wt).sum(1)
        return grad

    @staticmethod
    def estimate_sne(
        matrix: np.array,
        p: np.array,
        rng: np.random.Generator,
        num_iters: int,
        q_fn: callable,
        grad_fn: callable,
        learning_rate: float,
        momentum: float,
    ) -> np.array:
        """Estimate the low-dimensional representation using t-SNE.

        This static method estimates the low-dimensional representation of data
        points using t-SNE.
        It iteratively updates the low-dimensional data points (Y) to minimize
        the t-SNE loss.

        Args:
            matrix: The input data matrix.
            p: The joint probability distribution matrix.
            rng (np.random.Generator): A random number generator for initialization.
            num_iters: The number of optimization iterations.
            q_fn: A function for computing the joint probability distribution Q.
            grad_fn: A function for computing the gradient of the t-SNE loss.
            learning_rate: The learning rate for gradient descent.
            momentum: The momentum parameter for gradient descent.

        Returns:
            The low-dimensional representation of data points.

        """
        y = rng.normal(0.0, 0.0001, [matrix.shape[0], 2])
        if momentum:
            y_m2 = y.copy()
            y_m1 = y.copy()
        for i in range(num_iters):
            q, distances = q_fn(y)
            grads = grad_fn(p, q, y, distances)
            y = y - learning_rate * grads
            if momentum:
                y += momentum * (y_m1 - y_m2)
                y_m2 = y_m1.copy()
                y_m1 = y.copy()
        return y

    def fit_transform(self, matrix: np.array) -> np.array:
        """Fit the t-SNE model to the input data and transform it to
        low-dimensional space.

        This method performs both the training (fitting) and transformation of
        the input data into a low-dimensional space using t-SNE.
        It calculates the joint probability distribution matrix, optimizes the
        low-dimensional representation, and returns the
        transformed data.

        Args:
            matrix: The input data matrix.

        Returns:
            The low-dimensional representation of the input data.

        """
        matrix = np.array(matrix)
        p = self.p_joint(matrix, self.target_perplexity)
        rng = np.random.RandomState(self.seed)
        y = self.estimate_sne(
            matrix,
            p,
            rng,
            self.n_iter,
            q_fn=self.q_tsne if self.tse else self.q_joint,
            grad_fn=self.tsne_grad if self.tse else self.symmetric_sne_grad,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
        )
        return y