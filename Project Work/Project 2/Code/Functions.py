import numpy as np
import autograd.numpy as np 




def dataset(n=100, seed=42):
    #Generate a dataset based on the Runge's function with added Gaussian noise.
    np.random.seed(seed)
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




# Activation functions and their derivatives
def ReLU(z):
    return np.where(z > 0, z, 0)

def ReLU_der(z):
    return np.where(z > 0, 1, 0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_der(z):
    return sigmoid(z) * (1 - sigmoid(z))


def mse(predict, target):
    return np.mean((predict - target) ** 2)

def der_mse(predict, target):
    return 2 * (predict - target) / target.size


def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]

def der_softmax(predict, target):
    """Compute the derivative of the softmax function for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    return predict - target

#
def cross_entropy(predict, target):
    return np.sum(-target * np.log(predict))


def cost_batch(layers, input, activation_funcs, target):
    _, _, predict = feed_forward_saver_batch(input, layers, activation_funcs)
    return cross_entropy(predict, target)



# Neural Network functions


def feed_forward_saver_batch(input, layers, activation_funcs):
    layer_inputs = []
    zs = []
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = a @ W + b
        a = activation_func(z)

        zs.append(z)

    return layer_inputs, zs, a


def create_layers_batch(network_input_size, layer_output_sizes, seed=42):
    np.random.seed(seed)
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.rand(i_size, layer_output_size)
        b = np.random.rand(1,layer_output_size)
        layers.append((W, b))

        i_size = layer_output_size
    return layers

def feed_forward_saver(input, layers, activation_funcs):
    layer_inputs = []
    zs = []
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = W @ a + b
        a = activation_func(z)

        zs.append(z)

    return layer_inputs, zs, a

def cost(layers, input, activation_funcs, target):
    predict = feed_forward_saver(input, layers, activation_funcs)
    return mse(predict, target)

def backpropagation_batch(
    input, layers, activation_funcs, target, activation_ders, cost_der=der_mse
):
    layer_inputs, zs, predict = feed_forward_saver_batch(input, layers, activation_funcs)

    layer_grads = [() for layer in layers]

    # We loop over the layers, from the last to the first
    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]

        if i == len(layers) - 1:
            # For last layer we use cost derivative as dC_da(L) can be computed directly
            dC_da = cost_der(predict, target)
        else:
            # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
            (W, b) = layers[i + 1]
            dC_da = dC_dz @ W.T

        dC_dz = dC_da * activation_der(z)
        dC_dW = layer_input.T @ dC_dz 
        dC_db = dC_dz

        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads