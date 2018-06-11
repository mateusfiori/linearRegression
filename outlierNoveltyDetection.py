from sklearn.datasets import load_boston
from scipy.stats import iqr
import matplotlib.pyplot as plt
import numpy as np

boston = load_boston()

X = boston.data
y = boston.target

