# Extra functions for use in ex3.py

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import multiprocessing as mp

def displayData(X):

    fig, ax = plt.subplots(10,10,sharex=True,sharey=True)
    img_num = 0
    for i in range(10):
        for j in range(10):
            img = X[img_num,:].reshape(20,20).T
            ax[i][j].imshow(img,cmap='gray')
            img_num += 1

    return (fig, ax)

def displayImage(im):

    fig2, ax2 = plt.subplots()
    image = im.reshape(20,20).T
    ax2.imshow(image,cmap='gray')
    
    return (fig2, ax2)

def sigmoid(z):
    
    return 1.0/(1 +  np.e**(-z))

def lrCostFunction(theta,X,y,reg_param):

    m = len(y) 

    J =((np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-
       (1-y)*(np.log(1-sigmoid(np.dot(X,theta))))))/m +
       (reg_param/m)*np.sum(theta**2))

    grad_0 = (np.sum((sigmoid(np.dot(X,theta))-y)[:,None]*X,axis=0)/m)

    grad_reg = grad_0 + (reg_param/m)*theta

    grad_reg[0] = grad_0[0] 

    return (J,grad_reg)

def oneVsAll(X, y, num_labels, reg_param):

    n = np.size(X,1)
    theta = np.zeros((n,num_labels))

    def findOptParam(p_num):
        outcome = np.array(y == p_num).astype(int)
        initial_theta = theta[:,p_num]
        results = minimize(lrCostFunction,
			   initial_theta,
                           method='Newton-CG',
                           args=(X,outcome,reg_param),
                           jac=True,
		           tol=1e-6,
                           options={'maxiter':400,
                                    'disp':True})
        theta[:,p_num] = results.x
    
    
    for digit in range(10):
        findOptParam(digit)
    
    return theta


def predictOneVsAllAccuracy(est_theta,X):

    probs = np.dot(X,est_theta)
    predict = np.argmax(probs,axis=1)
    
    return predict


def predict(theta1,theta2,X):
    m = len(X)

    if np.ndim(X) == 1:
        X = X.reshape((-1,1))

    D1 = np.hstack((np.ones((m,1)),X))

    hidden_pred = np.dot(D1,theta1.T) 
    
    ones = np.ones((len(hidden_pred),1)) 
    hidden_pred = sigmoid(hidden_pred)
    hidden_pred = np.hstack((ones,hidden_pred))
    
    output_pred = np.dot(hidden_pred,theta2.T)   
    output_pred = sigmoid(output_pred)
    # Get predictions
    p = np.argmax(output_pred,axis=1)
    
    return p
