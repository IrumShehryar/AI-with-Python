import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler

df = pd.read_csv('../AI with Python/Admission_Predict.csv',skiprows =0, delimiter =',')
print(df)

x =df[['CGPA','GRE Score']]
y= df['Chance of Admit ']
x_train, x_test, y_train,y_test = train_test_split(x,y, random_state=30,test_size= 0.2 )
knn = neighbors.KNeighborsRegressor(n_neighbors= 5)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print(f' r2 score for knn {knn.score(x_test,y_test)}')

x_train_norm = MinMaxScaler().fit_transform(x_train)
x_test_norm = MinMaxScaler().fit_transform(x_test)

knn = neighbors.KNeighborsRegressor(n_neighbors= 5)
knn.fit(x_train_norm,y_train)
y_pred2 = knn.predict(x_test_norm)
print(f' r2 score for normalization {knn.score(x_test_norm,y_test)}')

x_train_std = StandardScaler().fit_transform(x_train)
x_test_std  = StandardScaler().fit_transform(x_test)
knn.fit(x_train_std,y_train)
y_pred3 = knn.predict(x_test_std)
print(f' r2 score for standardization {knn.score(x_test_std,y_test)}')

