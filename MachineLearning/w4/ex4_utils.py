# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def displayData(X):

    num_plots = int(np.size(X,0)**.5)
    fig, ax = plt.subplots(num_plots,num_plots,sharex=True,sharey=True)
    img_num = 0
    for i in range(num_plots):
        for j in range(num_plots):

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

def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))

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

    p = np.argmax(output_pred,axis=1)
    
    return p

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels,
                   X,y,reg_param):

    m = len(y) 

    theta1 = nn_params[:(hidden_layer_size * 
			           (input_layer_size + 1))].reshape((hidden_layer_size, 
							                             input_layer_size + 1))
  
    theta2 = nn_params[-((hidden_layer_size + 1) * 
                          num_labels):].reshape((num_labels,
					                             hidden_layer_size + 1))
   
    init_y = np.zeros((m,num_labels)) 
 
    for i in range(m):
        init_y[i][y[i]] = 1

    ones = np.ones((m,1)) 
    d = np.hstack((ones,X))
 
    cost = [0]*m

    D1 = np.zeros_like(theta1)
    D2 = np.zeros_like(theta2)
    for i in range(m):

        a1 = d[i][:,None] 
        z2 = np.dot(theta1,a1) 
        a2 = sigmoid(z2) 
        a2 = np.vstack((np.ones(1),a2)) 
        z3 = np.dot(theta2,a2) 
        h = sigmoid(z3) 
        a3 = h 

        cost[i] = (np.sum((-init_y[i][:,None])*(np.log(h)) -
	              (1-init_y[i][:,None])*(np.log(1-h))))/m

        d3 = a3 - init_y[i][:,None]
        d2 = np.dot(theta2.T,d3)[1:]*(sigmoidGradient(z2))
	

        D1 = D1 + np.dot(d2,a1.T) 
        D2 = D2 + np.dot(d3,a2.T) 

    reg = (reg_param/(2*m))*((np.sum(theta1[:,1:]**2)) + 
	      (np.sum(theta2[:,1:]**2)))

    grad1 = (1.0/m)*D1 + (reg_param/m)*theta1
    grad1[0] = grad1[0] - (reg_param/m)*theta1[0]
    
    grad2 = (1.0/m)*D2 + (reg_param/m)*theta2
    grad2[0] = grad2[0] - (reg_param/m)*theta2[0]
    
    # Append and unroll gradient
    grad = np.append(grad1,grad2).reshape(-1)
    final_cost = sum(cost) + reg

    return (final_cost, grad)



def randInitializeWeights(L_in,L_out):

    randWeights = np.random.uniform(low=-.12,high=.12,
                                    size=(L_in,L_out))
    return randWeights

def debugInitializeWeights(fan_in, fan_out):
    W = np.zeros((fan_out,fan_in + 1))
    W = np.array([np.sin(w) for w in 
                 range(np.size(W))]).reshape((np.size(W,0),np.size(W,1)))
    
    return W

def computeNumericalGradient(J,theta):

    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    tol = 1e-4
    
    for p in range(len(theta)):
    
        perturb[p] = tol
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
	
        # Compute numerical gradient
        numgrad[p] = (loss2 - loss1)/(2 * tol)
        perturb[p] = 0

	
    return numgrad

def checkNNGradients(reg_param):

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    Theta1 = debugInitializeWeights(hidden_layer_size,input_layer_size)
    Theta2 = debugInitializeWeights(num_labels,hidden_layer_size)

    X = debugInitializeWeights(input_layer_size - 1, m)

    y = [(i % num_labels) for i in range(m)]
    nn_params = np.append(Theta1,Theta2).reshape(-1)

    # Compute Cost
    cost, grad = nnCostFunction(nn_params,
                                input_layer_size,
                				hidden_layer_size,
                				num_labels,
                				X, y, reg_param)

    def reduced_cost_func(p):
        """ Cheaply decorated nnCostFunction """
        return nnCostFunction(p,input_layer_size,hidden_layer_size,num_labels,
                              X,y,reg_param)[0]

    numgrad = computeNumericalGradient(reduced_cost_func,nn_params)

    # Check two gradients
    np.testing.assert_almost_equal(grad, numgrad)

    return


