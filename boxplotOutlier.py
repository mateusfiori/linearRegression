import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

boston = load_boston()

X = boston.data
y = boston.target

plt.boxplot(X)
plt.show()
