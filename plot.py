import matplotlib.pyplot as plt
import numpy as np
import linearRegressionViews as r2TestSize
import linearRegressionPCA as r2PCA

#plot R2 x testSize
x = []
i = 0.1

#laço que cria uma array com o as labels para cada barra
while i < 0.5:
    x.append(str(round(i, 2)))
    i += 0.05

x1 = np.arange(x.__len__())

plt.subplot(2, 1, 1)
plt.title('R2 x testSize')
plt.xlabel('test_size')
plt.ylabel('R2 Score')
plt.bar(x1, r2TestSize.vecR2testSize)
plt.xticks(x1, x)
plt.ylim(0.5, 1.0)

#plot R2 x n_components
x = []
i = 13

# laço que cria uma array com o as labels para cada barra
while i > 1:
    x.append(str(round(i, 2)))
    i -= 1

x1 = np.arange(x.__len__())

plt.subplot(2, 1, 2)
plt.title('PCA')
plt.xlabel('Número de features')
plt.ylabel('R2 Score')
plt.plot(x1, r2PCA.vecR2components)
plt.xticks(x1, x)

plt.tight_layout(pad=2.0)
plt.show()
