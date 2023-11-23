import numpy as np


def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    # Initialize the eigenvector
    eigenvector = np.random.rand(data.shape[0])

    # Perform power method iterations
    for _ in range(num_steps):
        # Multiply the matrix with the eigenvector
        eigenvector = data.dot(eigenvector)

        # Normalize the eigenvector
        eigenvector /= np.linalg.norm(eigenvector)

    # Estimate the eigenvalue
    eigenvalue = eigenvector.dot(data.dot(eigenvector))

    return eigenvalue, eigenvector
