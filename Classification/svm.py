import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv('../AI with Python/iris.csv')

x = df.drop('species',axis=1)
y= df['species']
x_train, x_test, y_train,y_test = train_test_split(x,y, random_state=20,test_size= 0.2 )

svclassifier = SVC(kernel = 'linear')
svclassifier.fit(x_train,y_train)
y_pred1 = svclassifier.predict(x_test)
confusion_matrix1 = confusion_matrix(y_test,y_pred1)
print(f'Confusion matrix for SVM \n:{confusion_matrix1}')
print(classification_report(y_test,y_pred1))

svclassifier = SVC(kernel = 'poly', degree =2)
svclassifier.fit(x_train,y_train)
y_pred2 = svclassifier.predict(x_test)
confusion_matrix2 = confusion_matrix(y_test,y_pred2)
print(f'Confusion matrix for polynomial \n:{confusion_matrix2}')
print(classification_report(y_test,y_pred2))

svclassifier = SVC(kernel = 'rbf')
svclassifier.fit(x_train,y_train)
y_pred3 = svclassifier.predict(x_test)
confusion_matrix3 = confusion_matrix(y_test,y_pred3)
print(f'Confusion matrix for RBF \n:{confusion_matrix3}')
print(classification_report(y_test,y_pred3))


