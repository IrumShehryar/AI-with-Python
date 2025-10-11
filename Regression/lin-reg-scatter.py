import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("../AI with Python/linreg_data.csv", skiprows=0,names=["x","y"] )
#x= pd.read_csv("../AI with Python/linreg_data.csv")
xpd = df["x"]
ypd = df["y"]
n = xpd.size
#print(df)

#print(xbar)
#print(ybar)
product_xy= xpd*ypd
upper_term1= np.sum(product_xy)
print("summation_xy",upper_term1)
#print("product_xy",product_xy)
xbar= np.mean(xpd)
ybar= np.mean(ypd)

upper_term2= n*(xbar*ybar)
print("upper_term2",upper_term2)

numerator= upper_term1 - upper_term2
print("Numerator =",numerator)

x_square=np.square(xpd)
print("square of x =", x_square)
term3= np.sum(x_square)
print("term3 =", term3)

xbar_square = np.square(xbar)
term4 = n * xbar_square

print("term4 =", term4)
denominator = term3-term4
print("denominator",denominator)

b = numerator/denominator
print("slope b=",b)
#y= bx+ a
a = ybar - (b*xbar)
print("y intercept=",a)
#x_square=
plt.scatter(xpd,ypd, color='blue',label='Data points')
plt.scatter(xbar,ybar,color = 'red')

x= np.linspace(0,2,100)
y= a+b*x
plt.plot(x,y, linestyle="-",label="y = a+bx")

plt.show()
yhat= a + b * xpd
print("yhat", yhat)

ydiff= ypd - yhat
MAE= 1/n * np.sum( np.abs(ydiff))
print("MAE", MAE)

MSE = 1/n * np.sum(ydiff ** 2)
print("MSE", MSE)

y_mean_diff = (ypd- ybar)

R_square = 1 - (np.sum(ydiff ** 2)) / (np.sum(y_mean_diff ** 2))
print("R Square",R_square)

RMSE= np.sqrt((1/n) * np.sum(ydiff ** 2))
print("RMSE", RMSE)





