import matplotlib.pyplot as plt
import numpy as np

"""x= [2021,2022,2023,2024]
y= [100,200,300,400,500]"""

"""
plt.barh(['2021','2022','2023'],[120000,125000,130000],color='red')
plt.xlabel('years')
plt.ylabel('Sales')
plt.title('Sales record')
#plt.plot(x,y,color="red",marker="*",linestyle="--",linewidth=3)

plt.show()
"""
x=np.linspace(0,7,100)
y=np.sin(x)

plt.xlabel('x')
plt.ylabel('y')
plt.title('picture of sin graph')
plt.plot(x,y)
plt.show()
plt.subplot(1,2,1)
plt.plot(x,y)
plt.title('first subplot')
plt.subplot(1,2,2)
plt.plot(x,2*y)
plt.title("second subplot")
plt.show()
print(x)
print(y)