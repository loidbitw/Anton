#!/usr/bin/env python

"""
Machine Learning Online Class - Exercise 1: Linear Regression
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from ex1_utils import * 

# ==================== Part 1: Basic Function ====================
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')

print(warmUpExercise()) 
input('Program paused. Press enter to continue.\n')


# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = pd.read_csv("ex1data1.txt",names=["X","y"])
x = np.array(data.X)[:,None] 
y = np.array(data.y) 
m = len(y) 

# Plot Data
fig = plotData(x,y)
fig.show()

input('Program paused. Press enter to continue.\n')

## =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...\n')

ones = np.ones_like(x)
X = np.hstack((ones,x))
theta = np.zeros(2) 

iterations = 1500
alpha = 0.01

computeCost(X, y, theta)

theta, hist = gradientDescent(X, y, theta, alpha, iterations)

print('Theta found by gradient descent: ')
print(theta[0],"\n", theta[1])

plt.plot(x,y,'rx',x,np.dot(X,theta),'b-')
plt.legend(['Training Data','Linear Regression'])
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5],theta) 
print('For population = 35,000, we predict a profit of ', predict1*10000)

predict2 = np.dot([1, 7],theta)
print('For population = 70,000, we predict a profit of ', predict2*10000)

input('Program paused. Press enter to continue.\n');


## ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((len(theta0_vals),len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i],theta1_vals[j]])
        J_vals[i][j] = computeCost(X,y,t)

fig = plt.figure()
ax = plt.subplot(111,projection='3d')
Axes3D.plot_surface(ax,theta0_vals,theta1_vals,J_vals,cmap=cm.coolwarm)
plt.show()

fig = plt.figure()
ax = plt.subplot(111)
plt.contour(theta0_vals,theta1_vals,J_vals) 

