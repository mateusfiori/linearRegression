import matplotlib.pyplot as plt
import numpy as np
import linearRegressionViews as r2TestSizeLR
import ridgeRegression as r2TestSizeRidge
import lassoRegression as r2TestSizeLasso


#Esse script ira plotar os R2 scores de 3 diferentes regressores
#Linear Regression, Ridge Regression, Lasso Regression

#plot R2 x testSize
x = []
i = 0.1

#laço que cria uma array com o as labels para cada barra
while i < 0.5:
    x.append(str(round(i, 2)))
    i += 0.05

x1 = np.arange(x.__len__())

plt.figure(figsize=(7, 8))

plt.subplot(3, 1, 1)
plt.title('Linear Regression (testSize x R2 Score)')
plt.xlabel('test_size')
plt.ylabel('R2 Score')
plt.plot(x1, r2TestSizeLR.vecR2testSize)
plt.xticks(x1, x)
plt.ylim(0.1, 1.0)

plt.subplot(3, 1, 2)
plt.title('Ridge Regression (testSize x R2 Score)')
plt.xlabel('test_size')
plt.ylabel('R2 Score')
plt.plot(x1, r2TestSizeRidge.vecR2testSize)
plt.xticks(x1, x)
plt.ylim(0.1, 1.0)

plt.subplot(3, 1, 3)
plt.title('Lasso Regression (testSize x R2 Score)')
plt.xlabel('test_size')
plt.ylabel('R2 Score')
plt.plot(x1, r2TestSizeLasso.vecR2testSize)
plt.xticks(x1, x)
plt.ylim(0.1, 1.0)

plt.tight_layout(pad=1.0)
plt.show()
