import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.metrics
from sklearn.datasets import load_diabetes, load_wine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

data = load_wine(as_frame =True)
print(data.keys())
print(data.DESCR)
df = data['frame']
print(df)

plt.hist(df["target"],25)
plt.xlabel("target")
plt.show()

sns.heatmap(data=df.corr().round(2),annot=True)
plt.show()

plt.subplot(1,2,1)
plt.scatter(df['alcalinity_of_ash'],df['target'])
plt.xlabel('alcalinity_of_ash ')
plt.ylabel('target')
plt.subplot(1,2,2)
plt.scatter(df['nonflavanoid_phenols'],df['target'])
plt.xlabel('nonflavanoid_phenols')
plt.ylabel('target')
plt.show()
x = pd.DataFrame(df[['alcalinity_of_ash','nonflavanoid_phenols']], columns=['alcalinity_of_ash','nonflavanoid_phenols'])
y= df['target']

x_train, x_test,y_train, y_test = train_test_split(x,y ,random_state =5,test_size=0.4)
lm = LinearRegression()
lm.fit(x_train,y_train)
y_train_predict= lm.predict(x_train)
print("Root Mean absolute Error:", root_mean_squared_error(y_train_predict,y_train))
print("Root square Error:",r2_score(y_train,y_train_predict))

y_test_predict = lm.predict(x_test)
print("Root Mean absolute Error:", root_mean_squared_error(y_test_predict,y_test))
print("Root square Error:",r2_score(y_test,y_test_predict))