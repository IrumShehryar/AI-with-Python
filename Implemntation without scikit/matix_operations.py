import numpy as np

a= np.array([[1,2],[3,4]])
b= np.array([[5,6],[7,8]])
print(a,b)

print("sum=",a+b)
print("difference=",a-b)
print("product=",a*b)
print("square=",a**2)
print("division=",a/b)
print("matrice multiplication=",np.matmul(a,b))