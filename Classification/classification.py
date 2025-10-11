import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv('../AI with Python/exams.csv', skiprows = 0 , delimiter =',')
print(df)

x= df.iloc[:,0:2] # x =
y = df.iloc[:,-1]

admit_yes = df.loc[y==1]
admit_no = df.loc[y == 0]

plt.scatter(admit_no.iloc[:,0],admit_no.iloc[:,1],label = 'admit no')
plt.scatter(admit_yes.iloc[:,0], admit_yes.iloc[:,1],label ='admit yes')
plt.xlabel('exam 1')
plt.ylabel('exam 2')
plt.legend()
plt.show()


x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 5, test_size= 0.2)
Lr= LogisticRegression()
Lr.fit(x_train,y_train)
y_train_predict= Lr.predict(x_test)
metrics.ConfusionMatrixDisplay.from_estimator(Lr, x_test, y_test)
cnf_matrix = metrics.confusion_matrix(y_test,y_train_predict)
print(cnf_matrix)
plt.show()

precision = metrics.precision_score(y_test, y_train_predict)
accuracy = metrics.accuracy_score(y_test, y_train_predict)
recall = metrics.recall_score(y_test, y_train_predict)
print(f'precision = {precision}, accuracy = {accuracy} ,recall = {recall}')
"""
pred_yes_correct = cnf_matrix.iloc[:,0]
print(pred_yes_correct)

pred_yes_incorrect =
pred_no_correct =
pred_no_incorrect =

plt.scatter()
"""