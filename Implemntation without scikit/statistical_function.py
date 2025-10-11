import numpy as np

A= np.array([[1,2,3],[4,5,6]])
np.sum(A)

print("sum of rows is",np.sum(A,0))
print("sum f columns is",np.sum(A,1))

print("product", np.prod(A) )
print("product of rows",np.prod(A,0))
print("product of columns",np.prod(A,1))

print("mean of array is",np.mean(A))
print("mean of rows  is",np.mean(A,0))
print("mean of col is",np.mean(A,1))

print("median of array is",np.median(A))
print("median of rows is",np.median(A,0))
print("median of columns",np.median(A,1))