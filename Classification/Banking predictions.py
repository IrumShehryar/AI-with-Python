"""
_______________________________________________________________________________________________________________________________
4) Produce a heat map of correlation coefficients for all variables in df3. Describe the
amount of correlation between the variables in your own words.

Ans: The heatmap shows that most correlations are close to zero, so relationships between variables are generally weak and no
single feature strongly drives the outcome.
for instance:
poutcome_success has a correlation of +0.30 with y, which is a weak positive relationship. This means that when a person’s
previous campaign outcome was successful, they are more likely to say “yes” in the current campaign. Similarly,
poutcome_failure has a correlation of −0.25 with y, which is a weak negative relationship. This means that when a person’s
previous campaign outcome was a failure, they are less likely to say “yes” in the current campaign.
_________________________________________________________________________________________________________________________________

10) Compare the results between the two models.

OBSERVATION:

Logistic Regression:

Confusion matrix [[996, 9], [107, 19]]
TN: 996 / (996 + 9) = 996/1005 actual negatives caught
TP: 19 / (19 + 107) = 19/126 actual positives caught
Accuracy: 0.897.
Precision (class 1): 19 / (19 + 9) = 0.68 which means when it predicts “yes,” it’s right 68% of the time.
Recall (class 1): 19 / (19+107) = 0.15 which means it finds only about 15% of the actual “yes” cases.
Summary: high accuracy and decent precision, but very low recall for “yes.”

k-NN (k = 3)
Precision (class 1): 0.28.
Recall (class 1): 0.09 which means  it misses about 91% of actual “yes” cases.
Summary: at k=3, k-NN is worse than logistic on both precision and recall for the positive class.

For Different values of K
Checking for different values of k reveals that minimum error results when k is  between 7 to 18.
At k = 7 and 9, the test error is about 0.11, accuracy is 0.89 & precision for class 1 was 0.54.

CONCLUSION:

In this case Logistic Regression performs better as compared to KNN (k=3) on both precision and recall for the positive class, with similar accuracy.
For k=7 and 9, Logistic Regression is still slightly more accurate and has higher precision for the positive class

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import metrics

df = pd.read_csv('AI with Python/bank.csv' , delimiter= ';')
#print(df)

df2 = df[['y','job', 'marital','default', 'housing', 'poutcome']]
print(df2)
df3 = pd.get_dummies(df2,columns=['job','marital','default','housing','poutcome'])
df3['y'] = (df3['y'] == 'yes').astype(int)
print(df3)

"""
sns.heatmap(data=df3.corr().round(2),annot=True)
plt.show()
"""
# took this command from chatgpt as heatmap was unreadable
sns.heatmap(df3.corr(numeric_only=True).round(2), annot=True, fmt=".2f", cmap="vlag", center=0, square=True, annot_kws={"size":8}); plt.gcf().set_size_inches(11,9); plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()

X = df3.drop(columns=['y'])
y = df3['y']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state= 5 , test_size= 0.25)

Lr = LogisticRegression()
Lr.fit(X_train,y_train)
y_pred = Lr.predict(X_test)
metrics.ConfusionMatrixDisplay.from_estimator(Lr,X_test,y_test)
cnf_matrix =  metrics.confusion_matrix(y_test,y_pred)
print(f'cnf= {cnf_matrix}')
plt.show()

precision = metrics.precision_score(y_test, y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
print(f'precision = {precision}, accuracy = {accuracy} ,recall = {recall}')

kn = KNeighborsClassifier(n_neighbors = 3) # checked and calculated precision and recall for k=7 and k=9 for confirmation
kn.fit(X_train,y_train)
y_pred = kn.predict(X_test)
metrics.ConfusionMatrixDisplay.from_estimator(kn, X_test, y_test)
cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)
plt.show()
print(f'Classification report: {classification_report(y_test,y_pred)}')

# experimenting with different values of k the graph shows minimum errors between k = 7 to 18
error = []
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_train_predict = knn.predict(X_test)
    error.append(np.mean(y_train_predict != y_test))

plt.plot(range(1, 20), error, marker='o', markersize=10)
plt.xlabel('k')
plt.ylabel('error')
plt.show()