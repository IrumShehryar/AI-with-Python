import numpy as np
a= np.array([1,2,3,4])
arp= a.reshape(2,2)
print(arp)
#A = np.delete(arp,[1],1)

#print(A)
m,n=np.shape(arp)
print("m=",m,"n=",n)

for i in range(m):
    for j in range(n):
        print("The element",i,j,"is",arp[i][j])


for a in np.nditer(a):
    print(a)

b=arp
b[0,0]=18
print(b,arp)
b= arp.copy()
b[0,0]= 20
print("copied array",b," oroginal array",arp)

"""
z = np.zeros(10)
print(z)
z=np.ones((10,10))
print(z)
z=np.full((10,10),11)
print(z)

z1 = np.linspace(0,5,10)
print(z1)
z1= np.arange(5,2,-0.2)
print(z1)

z2 = np.random.random(10)
print(z2)

z2= np.random.randint(1,10,100)
print(z2)

z3 = np.random.randn(100)
print(z3)

z4 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
z4rp=z4.reshape(3,4)
print(z4rp)

z4rp = z4.reshape(2,1,6)
print(z4rp)

a = np.repeat([[1,1,1]],4,axis=0)
print(a)

a = np.array([1,2,3,4,5])
print(a)
print(a.ndim)

a = np.array([[1,2],[3,4]])
print(a)
print(a.ndim)

a = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(a)
print(a.ndim)
print(np.size(a))
print(np.shape(a))
print(a)

print("printing ",a[1][1][0])

a= np.array([[[[1,2,3],
               [4,5,6]],

            [[7,8,9],
             [10,11,12]],

              [[13,14,15],
               [16,17,18]]]])

print("printing value ",a[0,2,1,0])
print("shape",np.shape(a))
"""