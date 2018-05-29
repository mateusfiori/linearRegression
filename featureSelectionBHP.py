from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
import numpy as np
import matplotlib.pyplot as plt
import linearRegressionPCA as r2PCA

#O objetivo desse dataset é ilustrar o uso de feature_selection

#instanciacao do dataset Boston Housing Prices
boston = load_boston()

X = boston.data
y = boston.target

#instanciacao do regressor linear
lr = LinearRegression(normalize=True)

#f_regression é um teste feito e retorna um vetor com os scores de cada coluna e outro vetor de pvalues
#pvalue é a probabilidade do score f_regression ser maior que o valor empirico

#o return do f_regressopm é justamente os parametros do metodo SelectKBest
#metodo este que seleciona as K melhores features do dataset, i.e as features com maioreis variancias
#se uma coluna feature tem uma variacia muito baixa ela tende a se tornar irrelevante para o processo de aprendizado

vecFeatureSelection = []
n_components = X.shape[1]

while n_components > 1:

    #X_new = SelectKBest(f_regression, k=n_components).fit_transform(X, y)
    X_new = SelectKBest(mutual_info_regression, k=n_components).fit_transform(X, y)

    #divisao do conjuto de treinamento e conjunto de teste
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.15, random_state=42, shuffle=True)

    #Ajuste dos parametros
    lr.fit(X_train, y_train)

    #vetor com os valores preditos
    y_pred = lr.predict(X_test)

    print("n_components: {}\t\tR2: {}".format(n_components, r2_score(y_test, y_pred)))

    vecFeatureSelection.append(r2_score(y_test, y_pred))
    n_components -= 1

#plot R2 x n_components Select K Best
x = []
i = 13

# laço que cria uma array com o as labels para cada barra
while i > 1:
    x.append(str(round(i, 2)))
    i -= 1

x1 = np.arange(x.__len__())

plt.subplot(2, 1, 1)
plt.title('Select K-Best (f_regression)')
plt.xlabel('Número de features')
plt.ylabel('R2 Score')
plt.plot(x1, vecFeatureSelection)
plt.xticks(x1, x)

#plot R2 x n_components PCA

plt.subplot(2, 1, 2)
plt.title('PCA')
plt.xlabel('Número de features')
plt.ylabel('R2 Score')
plt.plot(x1, r2PCA.vecR2components)
plt.xticks(x1, x)

plt.tight_layout(pad=2.0)
plt.show()
