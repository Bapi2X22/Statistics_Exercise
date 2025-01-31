import numpy as np
import matplotlib.pyplot as plt


np.random.seed(seed=1234567)        # fix random seed
mu=0
sig=0.1
xl= np.random.normal(0,0.1,2000)
x=np.linspace(-3,3,1000)
pdf = (1/(np.sqrt(2*np.pi*(sig**2))))*np.exp(-((x-mu)**2)/(2*sig**2))
plt.hist(xl,bins=100,density=True)
plt.plot(x,pdf)
plt.show()
# for i in range(100):
#     x=np.random.normal(0,3)

# plt.hist(xb,bins=20,density=True)
# x=np.linspace(0,4,1000)
# y=2*np.exp(-2*x)
# plt.plot(x,y)
# plt.show()
# x= np.random.normal(0,3)
