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


# ReLU and Leaky ReLU
def ReLU(z):
    return np.where(z > 0, z, 0)

def der_ReLU(z):
    return np.where(z > 0, 1, 0)

def LeakyReLu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def der_LeakyReLu(X, alpha=0.01):
    return np.where(X > 0, 1, alpha)


# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def der_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Softmax

def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]


# Derivative of Softmax and Cross-Entropy combined
def der_softmax(predict, target):
    """Compute the derivative of the softmax function for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    return predict - target


#
# LOSS FUNCTIONS
#

def cross_entropy(predict, target):
    sample_losses = np.sum(-target * np.log(predict), axis=1)
    return np.mean(sample_losses)



# Mean Squared Error
def mse(predict, target):
    return np.mean((predict - target) ** 2)

def der_mse(predict, target):
    return 2 * (predict - target) 

#
# Not relevant for this project
#


def log_loss(y_true, y_pred, regularization=None, weights=None, lambda_reg=0.01):
    """
    Combines binary cross-entropy loss with L1 or L2 regularization.
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # Clipping it using epsilon to avoid log(0)
    m = len(y_true)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    if regularization == "L1" and weights is not None:
        loss += (lambda_reg / (2 * m)) * np.sum(np.abs(weights))
    elif regularization == "L2" and weights is not None:
        loss += (lambda_reg / (2 * m)) * np.sum(np.square(weights))
    return loss



#
# Optimization Algorithms
#


def Adam_init(Layers):
    # Initialize first and second moment estimates for Adam optimizer.
    m = [(np.zeros_like(w), np.zeros_like(b)) for w, b in Layers]
    v = [(np.zeros_like(w), np.zeros_like(b)) for w, b in Layers]
    return m, v

def Adam_update(Layers, grads, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # Update parameters using Adam optimization algorithm.
    new_Layers = []
    new_m = []
    new_v = []
    for (W, b), (dW, db), (m_w, m_b), (v_w, v_b) in zip(Layers, grads, m, v):
        # Update biased first moment estimate
        m_w = beta1 * m_w + (1 - beta1) * dW
        m_b = beta1 * m_b + (1 - beta1) * db

        # Update biased second raw moment estimate
        v_w = beta2 * v_w + (1 - beta2) * (dW ** 2)
        v_b = beta2 * v_b + (1 - beta2) * (db ** 2)

        # Compute bias-corrected first moment estimate
        m_w_hat = m_w / (1 - beta1 ** t)
        m_b_hat = m_b / (1 - beta1 ** t)

        # Compute bias-corrected second raw moment estimate
        v_w_hat = v_w / (1 - beta2 ** t)
        v_b_hat = v_b / (1 - beta2 ** t)

        # Update parameters
        W -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        b -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

        new_Layers.append((W, b))
        new_m.append((m_w, m_b))
        new_v.append((v_w, v_b))

    return new_Layers, new_m, new_v


# RMSprop Optimization

def RMSprop_init(Layers):
    # Initialize squared gradient moving averages for RMSprop optimizer.
    s = [(np.zeros_like(w), np.zeros_like(b)) for w, b in Layers]
    return s



def RMSprop_update(Layers, grads, s, learning_rate=0.001, beta=0.9, epsilon=1e-8):
    # Update parameters using RMSprop optimization algorithm.
    new_Layers = []
    new_s = []
    for (W, b), (dW, db), (s_w, s_b) in zip(Layers, grads, s):
        # Update the moving average of the squared gradients
        s_w = beta * s_w + (1 - beta) * (dW ** 2)
        s_b = beta * s_b + (1 - beta) * (db ** 2)

        # Update parameters
        W -= learning_rate * dW / (np.sqrt(s_w) + epsilon)
        b -= learning_rate * db / (np.sqrt(s_b) + epsilon)

        new_Layers.append((W, b))
        new_s.append((s_w, s_b))

    return new_Layers, new_s

#
# Neural Network functions
#

class NeuralNetwork:
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun,
        cost_der,
        regularization_type=None,
        regularization=None,
        method=None,
        seed=42
    ):
        # Standard initializer for the NeuralNetwork class.
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        # Regularization parameters
        self.regularization_type = regularization_type
        self.regularization = regularization

        # Seed for reproducibility
        self.seed = seed

        # Initialize layers
        self.layers = self._create_layers()
        
        # Initialize optimizer parameters
        if method == 'Adam':
            self.m, self.v = Adam_init(self.layers)
        elif method == 'RMSprop':
            self.rms = RMSprop_init(self.layers)

    def _create_layers(self):
        """
        Initializes weights and biases for all layers.
        """
        np.random.seed(self.seed)
        layers = []
        i_size = self.network_input_size
        for layer_output_size in self.layer_output_sizes:
            # Xavier initialization
            limit = np.sqrt(6.0 / (i_size + layer_output_size))
            W = np.random.uniform(-limit, limit, (i_size, layer_output_size))
            b = np.zeros((1, layer_output_size)) # Initialize biases to zero
            layers.append((W, b))
            i_size = layer_output_size
        return layers

    def predict(self, inputs):
        # Forward pass through the network to get predictions.
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a
    
    def cost(self, inputs, targets):
        # Computes the cost with or without regularization.
        base_cost = self.cost_fun(self.predict(inputs), targets)
        
        reg_cost = 0.0
        if self.regularization_type == "L2" and self.regularization is not None:
            for W, b in self.layers:
                reg_cost += (self.regularization / (2 * inputs.shape[0])) * np.sum(W ** 2)
        elif self.regularization_type == "L1" and self.regularization is not None:
            for W, b in self.layers:
                reg_cost += (self.regularization / (2 * inputs.shape[0])) * np.sum(np.abs(W))
        return base_cost + reg_cost

    def _feed_forward_saver(self, inputs):
        # Performs a forward pass and saves layer inputs and pre-activations.
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
        # Computes the gradient of the cost function w.r.t. the layers using backpropagation.
        layer_inputs, zs, predict = self._feed_forward_saver(inputs)
        layer_grads = [() for _ in self.layers]
        dC_dz = None # Initialize dC_dz

        n_samples = inputs.shape[0]

        # Loop over the layers, from the last to the first
        for i in reversed(range(len(self.layers))):
            layer_input, z = layer_inputs[i], zs[i]
            activation_der = self.activation_ders[i]

            (W,b) = self.layers[i]
            if i == len(self.layers) - 1:
                # For last layer, use cost derivative
                if self.activation_funcs[i].__name__ == "softmax":
                    dC_dz = self.cost_der(predict, targets)
                else:
                    dC_da = self.cost_der(predict, targets)
                    dC_dz = dC_da * activation_der(z)
            
            else:
                # For hidden layers, build on the previous gradient
                (W_next, b_next) = self.layers[i + 1]
                dC_da = dC_dz @ W_next.T
                dC_dz = dC_da * activation_der(z)

            dC_dW = layer_input.T @ dC_dz / n_samples
            
            # dC_db needs to be summed across the batch
            dC_db = np.sum(dC_dz, axis=0, keepdims=True) / n_samples

            if self.regularization_type == "L2" and self.regularization is not None:
                # Gradient is (lambda / n) * W
                dC_dW += (self.regularization / n_samples) * W   
            elif self.regularization_type == "L1" and self.regularization is not None:
                # Gradient is (lambda / n) * sign(W)
                dC_dW += (self.regularization / (2* n_samples)) * np.sign(W)
            
            layer_grads[i] = (dC_dW, dC_db)
        return layer_grads

    def update_weights(self, layer_grads):
        # Updates the weights and biases using the provided gradients.
        for i in range(len(self.layers)):
            W, b = self.layers[i]
            dC_dW, dC_db = layer_grads[i]
            W -= dC_dW
            b -= dC_db
            self.layers[i] = (W, b)
    
    def autograd_compliant_predict(self, layers, inputs):
        # Forward pass compatible with autograd for gradient computation.
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
    

def run_training_experiment(X_train, y_train, X_test, y_test,
                            n_epochs, batch_size, learning_rate,
                            activation_funcs, activation_ders,
                             cost_func, cost_der_func, layer_sizes,
                             method, 
                             lambda_reg=0.001, Regularization_type=None):
    """
    Trains a single NN with the given parameters and returns the test MSE.
    Has flexible choice of optimization method and regularization.
    Methods:
        GD, uses full-batch gradient descent.
        SGD, uses mini-batch stochastic gradient descent.
        RMSprop, uses mini-batch RMSprop optimization.
        Adam, uses mini-batch Adam optimization.
    """
    nn = NeuralNetwork(
        network_input_size=X_train.shape[1],
        layer_output_sizes=layer_sizes,
        activation_funcs=activation_funcs,
        activation_ders=activation_ders,
        cost_fun=cost_func,
        cost_der=cost_der_func, # Use original 2*(p-t)
        regularization_type=Regularization_type, # Use L2 for stability
        regularization=lambda_reg,
        method=method,
        seed=42
    )

    training_scores = []
    testing_scores = []
    t = 1 # Step counter for Adam
    best_score = 1e3
    # D. Training Loop
    for i in range(n_epochs):
        if method == 'GD' :
            grads = nn.compute_gradient(X_train, y_train)
            grads = [(dW * learning_rate, db * learning_rate) for dW, db in grads]
            nn.update_weights(grads)
        else:
            # Stochastic methods with mini-batches
            n_samples = X_train.shape[0]
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            # 2. Loop over mini-batches
            for start_idx in range(0, n_samples, batch_size):
                # Get the mini-batch
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                grads = nn.compute_gradient(X_batch, y_batch)
                update = []
                if method == 'SGD':
                    update = [(dW * learning_rate, db * learning_rate) for dW, db in grads]
                    nn.update_weights(update)
                elif method == 'RMSprop':
                    nn.layers, nn.rms = RMSprop_update(nn.layers, grads, nn.rms, learning_rate=learning_rate)
                elif method == 'Adam':
                    t += 1 # Increment timestep (one per batch)
                    nn.layers, nn.m, nn.v = Adam_update(nn.layers, grads, nn.m, nn.v, t, learning_rate=learning_rate)
        test_score = nn.cost(X_test, y_test)
        training_scores.append(nn.cost(X_train, y_train))
        testing_scores.append(test_score)
        if(best_score > test_score):
            best_score = test_score

    # E. Evaluate and Return

    return best_score, training_scores, testing_scores




import matplotlib.pyplot as plt


def Heatmap_with_labels(plot_mse, x,y,
            title='Heatmap', xlabel='Hidden Layers', ylabel='Nodes per Layer',
            barlabel='Test Mean Squared Error (log10)',
            resolution=100):
    fig, ax = plt.subplots(figsize=(8,6))
    
    # Use default indexing for imshow, don't set extent
    c = ax.imshow(plot_mse, aspect='auto', cmap='viridis')
    
    # Set the ticks to be at the center of the pixels (0, 1, etc.)
    ax.set_xticks(np.arange(len(x)))
    ax.set_yticks(np.arange(len(y)))
    
    # Set the *labels* for those ticks
    ax.set_xticklabels(x)
    ax.set_yticklabels(y)
    
    # Set labels and titles
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Add color bar
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.set_ylabel(barlabel, rotation=-90, va="bottom")
    
    # --- THIS IS THE CORRECTED ANNOTATION LOOP ---
    # Now (j, i) maps directly to the pixel coordinates
    threshold = 1
    for i in range(len(y)): # i = 0, 1 (rows, for nodes)
        for j in range(len(x)):   # j = 0, 1 (cols, for layers)
            val = plot_mse[i, j]
            if np.isfinite(val): # Only plot finite numbers
                color = "w" if val < threshold else "k" 
                # Use j for x-coordinate, i for y-coordinate
                text = ax.text(j, i, f"{val:.2f}",
                               ha="center", va="center", color=color, fontsize=12)
    plt.tight_layout()
    plt.show()