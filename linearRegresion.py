import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr as pearson

#existem duas principais divisoes do dataset Boston Housing Prices
#boston.data eh a matriz que contem todas as caracteristicas de cada item do dataset
#boston.target eh um vetor que contem de fato os precos de cada casa, o nosso objetivo

#instanciacao do dataset Boston Housing Prices
boston = load_boston()

X = boston.data
y = boston.target

#divisao do conjuto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)

#instanciacao do regressor linear
lr = LinearRegression(normalize=True)

#Ajuste dos parametros
lr.fit(X_train, y_train)

#vetor com os valores preditos
y_pred = lr.predict(X_test)

#Acuracia do estimator, regressor levando em consideração o conjunto de features para teste
#e o conjunto de alvos para teste
scoreLR = lr.score(X_test, y_test)

#Scores provenientes de Regression Metrics (sklearn.metrics)
R2 = r2_score(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
EVS = explained_variance_score(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
MeAE = median_absolute_error(y_test, y_pred)

#prints dos SCORES
print("Métricas:")
print()
print("R2 SCORE (lr): {}".format(scoreLR))
print("")
print("R2 SCORE (R2): {}".format(R2))
print("MEAN SQUARED ERROR (MSE): {}".format(MSE))
print("")
print("Explained Variance Score (EVS): {}".format(EVS))
print("Mean Absolut Error (MAE): {}".format(MAE))
print("Median Absolut Error (MeAE): {}".format(MeAE))




