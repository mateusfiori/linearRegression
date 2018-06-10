from sklearn.datasets import load_boston
from scipy.stats import iqr
import matplotlib.pyplot as plt
import numpy as np

boston = load_boston()

X = boston.data
y = boston.target

labelX = []
colArray = []
feat = 0

print(X.shape)

while feat < 13:
    colArray.append(iqr(X[:, feat]))
    labelX.append(str(feat))
    print("IQR (coluna {}: {}".format(feat, iqr(X[:, feat])))
    feat += 1

x1 = np.arange(labelX.__len__())

plt.title('IQR')
plt.xlabel('Index da coluna')
plt.ylabel('IQR value')
plt.scatter(x1, colArray)
plt.xticks(x1, labelX)

plt.show()