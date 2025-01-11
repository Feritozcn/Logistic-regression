import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv("read.csv")
X=data.drop("results",axis=1)

y=data["results"]
def sigmoid_function(X):
    return 1/(1+np.exp(-X))

n_samples,n_features=X.shape
def gradient_descent(X,y,l,epochs=1000):
    X=(X-X.mean()/X.std())
    weights=np.zeros(n_features)
    bias=0
    for i in range(epochs):
        y_pred=sigmoid_function(np.dot(X,weights)+bias)
        dw=(1/n_samples)*np.dot(X.T,(y_pred-y))
        db=(1/n_samples)*np.sum(y_pred-y)
        weights=weights - l*dw
        bias=bias - l*db
        if(i%100==0):
            print(f"epoch {i},cost={cost(X,y,weights,bias)}")
    return weights,bias

def cost(X, y, weights, bias):
    y_pred = sigmoid_function(np.dot(X, weights) + bias)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)# for avoiding floating point errors
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

weights,bias=gradient_descent(X.values,y.values,0.01)
print(weights,bias)

y_pred=sigmoid_function(np.dot(X.values,weights)+bias)
y_pred= [1 if pred>=0.5  else 0 for pred in y_pred]
print(y_pred)
# plt.plot(X["info1"],y_pred)
# plt.show()
