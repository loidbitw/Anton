
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from ex1_utils import *

## ================ Part 1: Feature Normalization ================

print('Loading data ...','\n')

print('Plotting Data ...','\n')

data = pd.read_csv("ex1data2.txt",names=["size","bedrooms","price"])
s = np.array(data.size)
r = np.array(data.bedrooms)
p = np.array(data.price)
m = len(r) 

s = np.vstack(s)
r = np.vstack(r)
X = np.hstack((s,r))


print('First 10 examples from the dataset: \n')
print(" size = ", s[:10],"\n"," bedrooms = ", r[:10], "\n")

input('Program paused. Press enter to continue.\n')


print('Normalizing Features ...\n')

X = featureNormalize(X)

X = np.hstack((np.ones_like(s),X))

## ================ Part 2: Gradient Descent ================

print('Running gradient descent ...\n')

alpha = 0.05
num_iters = 400

theta = np.zeros(3)

theta, hist = gradientDescent(X, p, theta, alpha, num_iters)

fig = plt.figure()
ax = plt.subplot(111)
plt.plot(np.arange(len(hist)),hist ,'-b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

print('Theta computed from gradient descent: \n')
print(theta,'\n')

normalized_specs = np.array([1,((1650-s.mean())/s.std()),((3-r.mean())/r.std())])
price = np.dot(normalized_specs,theta) 


print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ',
      price)

input('Program paused. Press enter to continue.\n')

## ================ Part 3: Normal Equations ================
print('Solving with normal equations...\n')

data = pd.read_csv("ex1data2.txt",names=["sz","bed","price"])
s = np.array(data.sz)
r = np.array(data.bed)
p = np.array(data.price)
m = len(r) # number of training examples

s = np.vstack(s)
r = np.vstack(r)
X = np.hstack((s,r))

X = np.hstack((np.ones_like(s),X))

theta = normalEqn(X, p)

print('Theta computed from the normal equations: \n')
print(theta)
print('\n')

price = np.dot([1,1650,3],theta) 


print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): \n',
       price)

