import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

#existem duas principais divisoes do dataset Boston Housing Prices
#boston.data eh a matriz que contem todas as caracteristicas de cada item do dataset
#boston.target eh um vetor que contem de fato os precos de cada casa, o nosso objetivo

#instanciacao do dataset Boston Housing Prices
boston = load_boston()

X = boston.data
y = boston.target

vecR2testSize = []
testSize = 0.1

print("")

# instanciacao do regressor linear
lr = LinearRegression(normalize=True)

#score R2 de acordo com o tamanho do conjunto de treinamento
while testSize < 0.5:

    #divisao do conjuto de treinamento e conjunto de teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42, shuffle=True)

    #Ajuste dos parametros
    lr.fit(X_train, y_train)

    #vetor com os valores preditos
    y_pred = lr.predict(X_test)

    #Scores provenientes de Regression Metrics (sklearn.metrics)
    R2 = r2_score(y_test, y_pred)

    print("Test size: {}\t\t\tR2 Score: {}".format(round(testSize, 2), R2))

    vecR2testSize.append(R2)
    testSize += 0.05

print("")
'''
x = []
i = 0.1

#laço que cria uma array com o as labels para cada barra
while i < 0.5:
    x.append(str(round(i, 2)))
    i += 0.05

x1 = np.arange(x.__len__())

plt.xlabel('test_size')
plt.ylabel('R2 Score')
plt.bar(x1, vecR2)
plt.xticks(x1, x)
plt.ylim(0.5, 1.0)
plt.show()


print("\nVetor de R2: {}".format(vecR2testSize))
'''