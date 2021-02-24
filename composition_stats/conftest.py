import numpy as np


def pytest_configure(config):
    np.set_printoptions(suppress=True, sign=' ')
