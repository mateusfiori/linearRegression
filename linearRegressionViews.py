import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#existem duas principais divisoes do dataset Boston Housing Prices
#boston.data eh a matriz que contem todas as caracteristicas de cada item do dataset
#boston.target eh um vetor que contem de fato os precos de cada casa, o nosso objetivo

#instanciacao do dataset Boston Housing Prices
boston = load_boston()

X = boston.data
y = boston.target

vecR2 = []
testSize = 0.1

print("")

while testSize < 0.5:

    #divisao do conjuto de treinamento e conjunto de teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42, shuffle=True)

    #instanciacao do regressor linear
    lr = LinearRegression(normalize=True)

    #Ajuste dos parametros
    lr.fit(X_train, y_train)

    #vetor com os valores preditos
    y_pred = lr.predict(X_test)

    #Scores provenientes de Regression Metrics (sklearn.metrics)
    R2 = r2_score(y_test, y_pred)

    print("Test size: {}\t\tR2 Score: {}".format(round(testSize, 2), R2))

    vecR2.append(R2)
    testSize += 0.05

print("\nVetor de R2: {}".format(vecR2))
