import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import numpy as np

#existem duas principais divisoes do dataset Boston Housing Prices
#boston.data eh a matriz que contem todas as caracteristicas de cada item do dataset
#boston.target eh um vetor que contem de fato os precos de cada casa, o nosso objetivo


#instanciacao do dataset Boston Housing Prices
boston = load_boston()

X = boston.data
y = boston.target

vecR2components = []
n_components = X.shape[1]

#score R2 de acordo com o numero de componentes
while n_components > 1:

    pca = PCA(n_components=n_components)
    newX = pca.fit_transform(X)

    #divisao do conjuto de treinamento e conjunto de teste
    X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size=0.15, random_state=42, shuffle=True)

    #instanciacao do regressor linear
    lr = LinearRegression(normalize=True)

    #Ajuste dos parametros
    lr.fit(X_train, y_train)

    #vetor com os valores preditos
    y_pred = lr.predict(X_test)

    #Scores provenientes de Regression Metrics (sklearn.metrics)
    R2 = r2_score(y_test, y_pred)

    print("n_components: \t{}\t\tR2 Score: {}".format(n_components, R2))

    vecR2components.append(R2)
    n_components -= 1

'''
x = []
i = 13

# laço que cria uma array com o as labels para cada barra
while i > 1:
    x.append(str(round(i, 2)))
    i -= 1

x1 = np.arange(x.__len__())

plt.xlabel('Número de features')
plt.ylabel('R2 Score')
plt.bar(x1, vecR2components)
plt.xticks(x1, x)
plt.show()


print("")
print(vecR2components)

'''