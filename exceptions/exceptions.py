class BaseException(Exception):
    """Base exception"""


class InvalidModelError(BaseException):
    """Raised when the choice of RNN model is invalid."""


class InvalidDatasetSelection(BaseException):
    """Raised when the choice of dataset is invalid."""

class InvalidNumNeighborsKNN(BaseException):
    """Raised when the number of neighbors for the KNN classifier is null"""

class InvalidClassifier(BaseException):
    """Raised when the choice of classifier is invalid"""