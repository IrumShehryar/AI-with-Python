import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('../AI with Python/iris.csv')

x= df.iloc[:,0:4] # x =
y = df.iloc[:,-1]

iris_setosa= df.loc[y =='Iris-setosa']
iris_versicolor= df.loc[y == 'Iris-versicolor']
iris_virginica = df.loc[y == 'Iris-virginica']


plt.scatter(iris_setosa.iloc[:,0],iris_setosa.iloc[:,1],label = 'Iris-setosa')
plt.scatter(iris_versicolor.iloc[:,0], iris_versicolor.iloc[:,1],label ='iris_versicolor')
plt.scatter(iris_virginica.iloc[:,0], iris_virginica.iloc[:,1],label ='iris_virginica')
plt.xlabel('features')
plt.ylabel(' catagory')
plt.legend()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 5, test_size= 0.2)
classifier=KNeighborsClassifier(n_neighbors= 5)
classifier.fit(x_train,y_train)
y_train_predict= classifier.predict(x_test)
metrics.ConfusionMatrixDisplay.from_estimator(classifier, x_test, y_test)
cnf_matrix = metrics.confusion_matrix(y_test,y_train_predict)
print(cnf_matrix)
plt.show()
print(f'Classification report: {classification_report(y_test,y_train_predict)}')

error = []
for k in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_train_predict = knn.predict(x_test)
    error.append(np.mean(y_train_predict!= y_test))
  

plt.plot(range(1,20), error,marker ='o',markersize = 10)
plt.xlabel('k')
plt.ylabel('error')
plt.show()