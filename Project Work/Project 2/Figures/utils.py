import numpy as np

#
# Setup functions
#

def dataset(n=100):
    #Generate a dataset based on the Runge's function with added Gaussian noise.
    np.random.seed(42)
    x = np.linspace(-1,1,n)
    y = 1/(1+25*x**2) 
    y = y.reshape(n,1) 
    y_noise = y + np.random.normal(0,0.1,size=(n,1))
    return x, y, y_noise

def polynomial_features(x, p, intercept=False):
    #Generate polynomial features up to degree p for input data x.
    n = len(x)
    k = 0
    if intercept:
        X = np.zeros((n, p + 1))
        X[:, 0] = 1
        k += 1
    else:
        X = np.zeros((n, p))
    for i in range(1, p +1):
        X[:, i + k-1] = x**i 
    return X

#
# Metrics
#

def MSE(y, y_pred):
    #Mean Squared Error
    return np.mean((y - y_pred)**2)

def R2(y, y_pred):
    #R2 score
    return 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))

#
# Regression Functions
#

# Ordinary Least Squares 
def OLS_parameters(X, y):
    # Ordinary Least Square analytical solution
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def Gradient_OLS(X, y,theta, eta=0.01, n=100):
    # Gradient solution for OLS
    return (2.0/n)*X.T @ (X @ theta-y)


# Ridge Regression
def Ridge_parameters(X, y, regularization=.001):
    # Analytical solution for Ridge regression
    return np.linalg.pinv(X.T @ X + regularization * np.identity(len(X.T))) @ X.T @ y

def Gradient_Ridge(X, y, theta, eta=0.01, lambda_param=0.01,n=100):
    # Gradient solution for Ridge regression
    return (2.0/n)*X.T @ (X @ theta-y) + 2*lambda_param*theta 


# Lasso Regression
def LassoIter(X, y, theta, eta=0.01, lambda_param=.01,n=100):
    # Iterative solution for Lasso regression
    grad_OLS = Gradient_OLS(X, y, eta=eta,theta=theta, n=n) # Finding the first part of cost function
    theta -= eta * grad_OLS
    theta =  np.sign(theta) * np.maximum(0, np.abs(theta) - eta * lambda_param) # Applying the soft thresholding operator
    return theta

