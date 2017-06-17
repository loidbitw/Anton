
import numpy as np
import matplotlib.pyplot as plt

def warmUpExercise():

    return np.eye(5)


def plotData(x, y):

    fig, ax = plt.subplots() # create empty figure and set of axes
    ax.plot(x,y,'rx',markersize=10)
    ax.set_xlabel("Population of City in 10,000s")
    ax.set_ylabel("Profit in $10,000s")

    return fig


def normalEqn(X,y):

    
    return np.dot((np.linalg.inv(np.dot(X.T,X))),np.dot(X.T,y))


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    
    m = len(y) 
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha/m)*np.sum((np.dot(X,theta)-y)[:,None]*X,axis=0)

        J_history[i] = computeCost(X, y, theta)
        print('Cost function has a value of: ', J_history[i])
    
    return (theta,J_History)


def gradientDescent(X, y, theta, alpha, num_iters):

    m = len(y) 
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha/m)*np.sum((np.dot(X,theta)-y)[:,None]*X,axis=0)

        J_history[i] = computeCost(X, y, theta)
        print('Cost function as a value of: ',J_history[i])
    
    return (theta, J_history)


def featureNormalize(X):

    return np.divide((X - np.mean(X,axis=0)),np.std(X,axis=0))


def computeCost(X, y, theta):

    m = len(y)
    
    J = (np.sum((np.dot(X,theta) - y)**2))/(2*m)

    return J

