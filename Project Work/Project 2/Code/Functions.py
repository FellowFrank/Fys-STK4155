import autograd.numpy as np 
from autograd import grad # Import grad for autograd_gradient




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
def linear(z):
    return z
def der_linear(z):
    return np.ones_like(z)

def ReLU(z):
    return np.where(z > 0, z, 0)

def der_ReLU(z):
    return np.where(z > 0, 1, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def der_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


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

def mse(predict, target):
    return np.mean((predict - target) ** 2)

def der_mse(predict, target):
    return 2 * (predict - target) / target.size



def cost_batch(layers, input, activation_funcs, target, costfunction=cross_entropy):
    _, _, predict = feed_forward_saver_batch(input, layers, activation_funcs)
    return costfunction(predict, target)



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

def cost(layers, input, activation_funcs, target, costfunction):
    predict = feed_forward_saver(input, layers, activation_funcs)
    return costfunction(predict, target)

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
        dC_db = np.sum(dC_dz, axis=0, keepdims=True) # had an error from earlier codes

        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads





class NeuralNetwork:
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun,
        cost_der,
        seed=42
    ):
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.seed = seed
        
        self.layers = self._create_layers()

    def _create_layers(self):
        """
        Initializes weights and biases for all layers.
        """
        np.random.seed(self.seed)
        layers = []
        i_size = self.network_input_size
        for layer_output_size in self.layer_output_sizes:
            W = np.random.normal(0, 1, (i_size, layer_output_size))
            b = np.zeros((1, layer_output_size)) # Initialize biases to zero
            layers.append((W, b))
            i_size = layer_output_size
        return layers

    def predict(self, inputs):
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a
    
    def cost(self, inputs, targets):
        return self.cost_fun(self.predict(inputs), targets)

    def _feed_forward_saver(self, inputs):
        layer_inputs = []
        zs = []
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a

    def compute_gradient(self, inputs, targets):
        """
        Performs backpropagation and computes gradients for a batch.
        """
        layer_inputs, zs, predict = self._feed_forward_saver(inputs)
        layer_grads = [() for _ in self.layers]
        dC_dz = None # Initialize dC_dz

        # Loop over the layers, from the last to the first
        for i in reversed(range(len(self.layers))):
            layer_input, z = layer_inputs[i], zs[i]
            activation_der = self.activation_ders[i]

            if i == len(self.layers) - 1:
                # For last layer, use cost derivative
                dC_da = self.cost_der(predict, targets)
            else:
                # For hidden layers, build on the previous gradient
                (W_next, b_next) = self.layers[i + 1]
                dC_da = dC_dz @ W_next.T

            dC_dz = dC_da * activation_der(z)
            dC_dW = layer_input.T @ dC_dz
            
            # dC_db needs to be summed across the batch
            dC_db = np.sum(dC_dz, axis=0, keepdims=True)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

    def update_weights(self, layer_grads):
        for i in range(len(self.layers)):
            W, b = self.layers[i]
            dC_dW, dC_db = layer_grads[i]
            W -= dC_dW
            b -= dC_db
            self.layers[i] = (W, b)
    
    # These last two methods are not needed in the project, but they can be nice to have! The first one has a layers parameter so that you can use autograd on it
    def autograd_compliant_predict(self, layers, inputs):
        a = inputs
        for (W, b), activation_func in zip(layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a
    
    def autograd_gradient(self, inputs, targets):
        """
        Computes the gradient of the cost function w.r.t. the layers
        using autograd.
        """
        
        # Define a cost function that takes 'layers' as the first argument
        def autograd_cost_for_grad(layers):
            # Use the autograd-compliant predict function
            predictions = self.autograd_compliant_predict(layers, inputs)
            # Use the cost function stored in self
            return self.cost_fun(predictions, targets)

        # Create the gradient function using autograd.grad
        # This function will differentiate autograd_cost_for_grad
        # with respect to its first argument (layers).
        gradient_calculator = grad(autograd_cost_for_grad)

        # Calculate and return the gradient for the current layers
        return gradient_calculator(self.layers)