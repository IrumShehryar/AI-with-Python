import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, ridge_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('../AI with Python/ridgereg_data.csv')
x = df[['x']]
y = df[['y']]
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=5 , test_size=0.2)
alphas = np.linspace(0,2,50)
print(alphas)
r2Values = []
for alp in alphas:
    rr = Ridge(alpha = alp)
    rr.fit(x_train, y_train)
    r2_test = r2_score(y_test,rr.predict(x_test))
    r2Values.append(r2_test)
plt.plot(alphas, r2Values)
plt.show()
best_r2 = max(r2Values)
print(best_r2)

idx = r2Values.index(best_r2)
best_alpha = alphas[idx]
print(idx,best_alpha)