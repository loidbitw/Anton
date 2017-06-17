
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ex2_utils import *


data = pd.read_csv('ex2data2.txt', names=['x1','x2','y'])
X = np.asarray(data[["x1","x2"]])
y = np.asarray(data["y"])

fig, ax = plotData(X, y)

ax.legend(['Pass', 'Fail'])

ax.set_xlabel('Microchip test 1')
ax.set_ylabel('Microchip test 2')

fig.show()

input('\nProgram paused. Press enter to continue.\n')

X = mapFeatureVector(X[:,0],X[:,1])

initial_theta = np.zeros(len(X[0,:]))

reg_param = 1.0

res = minimize(costFunctionReg,
	       initial_theta,
	       args=(X,y,reg_param),
	       tol=1e-6,
	       options={'maxiter':400,
			'disp':True})

theta = res.x

fig.clear()
fig, ax = plotDecisionBoundary(theta,X,y)


ax.legend(['Pass', 'Fail','Decision Boundary'])

ax.set_xlabel('Microchip test 1')
ax.set_ylabel('Microchip test 2')
ax.set_title('Lambda = 1')

fig.show()

input('\nProgram paused. Press enter to continue.\n')
