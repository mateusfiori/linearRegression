import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#existem duas principais divisoes do dataset Boston Housing Prices
#boston.data eh a matriz que contem todas as caracteristicas de cada item do dataset
#boston.target eh um vetor que contem de fato os precos de cada casa, o nosso objetivo

#instanciacao do dataset Boston Housing Prices
boston = load_boston()

X = boston['data']
y = boston['target']

vecR2testSize = []
testSize = 0.1

#instanciacao do regressor Lasso
lassor = Lasso()

while testSize < 0.5:

    #divisao do conjuto de treinamento e conjunto de teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)


    #Ajuste dos parametros
    lassor.fit(X_train, y_train)

    #vetor com os valores preditos
    y_pred = lassor.predict(X_test)

    vecR2testSize.append(r2_score(y_test, y_pred))
    testSize += 0.05


#Acuracia do estimator, regressor levando em consideração o conjunto de features para teste
#e o conjunto de alvos para teste
scoreLR = lassor.score(X_test, y_test)

#Scores provenientes de Regression Metrics (sklearn.metrics)
R2 = r2_score(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
EVS = explained_variance_score(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
MeAE = median_absolute_error(y_test, y_pred)

#prints dos SCORES
print("Métricas:")
print("")
print("R2 SCORE (R2): {}".format(R2))
print("MEAN SQUARED ERROR (MSE): {}".format(MSE))
print("")
print("Explained Variance Score (EVS): {}".format(EVS))
print("Mean Absolut Error (MAE): {}".format(MAE))
print("Median Absolut Error (MeAE): {}".format(MeAE))




