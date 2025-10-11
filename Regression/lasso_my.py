import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error  ,r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df = pd.read_csv('../AI with Python/diamonds.csv')
x = df[['carat', 'depth', 'table', 'x', 'y', 'z']]
y = df['price']
x_train,x_test,y_train, y_test = train_test_split(x,y,random_state=5 ,test_size= 0.2)
alphas = [0.1,0.2,0.3,0.4,0.5,1,2,3,4,5,6,7,8]
print('alphas',alphas)
r2Values = []
for alp in alphas:
    lr = linear_model.Lasso(alpha=alp)
    lr.fit(x_train, y_train)
    r2_test = r2_score(y_test, lr.predict(x_test))
    r2Values.append(r2_test)
plt.plot(alphas, r2Values)
plt.show()
best_r2 = max(r2Values)
print(best_r2)

idx = r2Values.index(best_r2)
best_alpha = alphas[idx]
print(f'best r2={best_r2}, best_alpha={best_alpha}')