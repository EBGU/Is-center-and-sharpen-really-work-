import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def softmax(x,t=1):
    inter = np.exp(x/t)
    p_x = inter / np.sum(inter,axis=1,keepdims=True)
    return p_x

def entropy(p_x):
    e_x = -np.sum(np.log(p_x)*p_x,axis=1)
    return e_x

def cross_entropy(p_x,p_y):
    h_x = -np.sum(np.log(p_y)*p_x,axis=1)
    return h_x

dim = 256
sample = 100000
ts = 0.1
tt = 0.04

#x = np.random.rand(sample,dim)
x = np.random.normal(1,0.1,(sample,dim))
y = x + np.random.normal(0,0.1,(sample,dim))

p_t = softmax(x,tt)
p_s = softmax(y,ts)

e = entropy(p_t)
h = cross_entropy(p_s,p_t)

reg = LinearRegression().fit(e.reshape(-1,1), (h-e).reshape(-1,1))
print(reg.coef_)

plt.scatter(e,h-e,s=1)
plt.show()