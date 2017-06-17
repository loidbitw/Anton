
# Imports
import numpy as np
import matplotlib.pyplot as plt


def plotData(X, y):


    pos = X[np.where(y==1)]
    neg = X[np.where(y==0)]

    fig, ax = plt.subplots()
    ax.plot(pos[:,0],pos[:,1],"k+",neg[:,0],neg[:,1],"yo")

    return (fig, ax)

def costFunction(theta,X,y):

    m = len(y) 
    
    # Cost function J(theta)
    J =(np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-
       (1-y)*(np.log(1-sigmoid(np.dot(X,theta)))))/m)

    # Gradient
    grad = (np.sum((sigmoid(np.dot(X,theta))-y)[:,None]*X,axis=0)/m)
  
    return (J, grad)

def sigmoid(z): 
    return 1.0/(1 +  np.e**(-z))


def predict(theta,X):

    return np.where(np.dot(X,theta) > 5.,1,0)

def mapFeatureVector(X1,X2):
    
    degree = 6
    output_feature_vec = np.ones(len(X1))[:,None]

    for i in range(1,7):
        for j in range(i+1):
            new_feature = np.array(X1**(i-j)*X2**j)[:,None]
            output_feature_vec = np.hstack((output_feature_vec,new_feature))
   
    return output_feature_vec


def costFunctionReg(theta,X,y,reg_param):

    m = len(y) 

    J =((np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-
       (1-y)*(np.log(1-sigmoid(np.dot(X,theta))))))/m +
       (reg_param/m)*np.sum(theta**2))

    grad_0 = (np.sum((sigmoid(np.dot(X,theta))-y)[:,None]*X,axis=0)/m)
    grad_reg = grad_0 + (reg_param/m)*theta
    grad_reg[0] = grad_0[0] 
    
    return J


def plotDecisionBoundary(theta,X,y):

    fig, ax = plotData(X[:,1:],y)

    u = np.linspace(-1,1.5,50)
    v = np.linspace(-1,1.5,50)
    z = np.zeros((len(u),len(v)))
    
    for i in range(len(u)):
        for j in range(len(v)):
            z[i][j] = np.dot(mapFeatureVector(np.array([u[i]]),
		      np.array([v[j]])),theta)
    
    ax.contour(u,v,z,levels=[0])

    return (fig,ax)



