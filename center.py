import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def softmax(x,t=1):
    inter = np.exp(x/t)
    p_x = inter / np.sum(inter,axis=2,keepdims=True)
    return p_x

def entropy(p_x):
    e_x = -np.sum(np.log(p_x)*p_x,axis=(1,2))
    return e_x

def cross_entropy(p_x,p_y):
    h_x = -np.sum(np.log(p_y)*p_x,axis=(1,2))
    return h_x

def center(x):
    x = x - np.mean(x,axis=1,keepdims=True)
    return x


dim = 256
batchSize = 256
sample = 1000
ts = 0.1 #try 0.3
tt = 0.1 #try 0.3

#x = np.random.rand(sample,batchSize,dim)

x = np.array([np.random.normal(1,i*1e-3+0.5,(batchSize,dim)) for i in range(sample)])
y = x + np.random.normal(0,0.1,(sample,batchSize,dim))
z = center(x)

p_t = softmax(z,tt)
p_s = softmax(y,ts)

e = entropy(p_t)
h = cross_entropy(p_s,p_t)

var = (np.sum(z**2,axis=(1))**0.5).mean(axis=1)

reg = LinearRegression().fit(var.reshape(-1,1), (h-e).reshape(-1,1))
print(reg.coef_)

plt.scatter(var,h-e,s=1)
plt.show()