"""Test utilities."""

from functools import wraps
import matplotlib.pyplot as plt


def plot_test(func):
    """Decorator for simple testing of plotting functions.

    Notes
    -----
    This decorator closes all plots prior to the test.
    After running the test function, it checks an axis was created with data.
    It therefore performs a minimal test - asserting the plots exists, with no accuracy checking.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        plt.close('all')

        func(*args, **kwargs)

        ax = plt.gca()
        assert ax.has_data()

    return wrapper


def pbar(func, *args, **kwargs):
    # Test dummy progress bar
    return func


class TestModel:
    def __init__(self):
        self.result = None

    def fit(self, *arrays):

        if len(arrays) == 2:
            _, y = arrays
        else:
            y = arrays[0]

        self.result = y.sum()

    def fit_transform(self, *arrays):

        if len(arrays) == 2:
            _, y = arrays
        else:
            y = arrays[0]

        return y.mean(axis=1)
