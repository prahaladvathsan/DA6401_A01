import numpy as np
import wandb
import argparse
from keras.datasets import fashion_mnist, mnist
from tqdm import tqdm


# Activation functions
class Activation:
    @staticmethod
    def identity(x, derivative=False):
        if derivative:
            return np.ones_like(x)
        return x
    
    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x, derivative=False):
        if derivative:
            return 1 - np.tanh(x) ** 2
        return np.tanh(x)
    
    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)


# Loss functions
class Loss:
    @staticmethod
    def mean_squared_error(y_true, y_pred, derivative=False):
        if derivative:
            return (y_pred - y_true)
        return 0.5 * np.mean(np.sum((y_pred - y_true) ** 2, axis=1))
    
    @staticmethod
    def cross_entropy(y_true, y_pred, derivative=False):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if derivative:
            return -y_true / y_pred + (1 - y_true) / (1 - y_pred)
        
        # For numerical stability and avoiding log(0)
        return -np.mean(np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=1))


# Optimizers

class Optimizer:
    def __init__(self, learning_rate=0.01, **kwargs):
        self.learning_rate = learning_rate
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def update(self, weights, gradients):
        raise NotImplementedError
    
    def apply_weight_decay(self, weights, gradients, weight_decay):
        if weight_decay > 0:
            return gradients + weight_decay * weights
        return gradients


class SGD(Optimizer):
    def update(self, weights, gradients):
        gradients = self.apply_weight_decay(weights, gradients, getattr(self, 'weight_decay', 0))
        return weights - self.learning_rate * gradients


class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, **kwargs):
        super().__init__(learning_rate, **kwargs)
        self.momentum = momentum
        self.velocity = {}  # Dictionary to store velocities for each unique weight matrix
    
    def update(self, weights, gradients):
        gradients = self.apply_weight_decay(weights, gradients, getattr(self, 'weight_decay', 0))
        
        # Create a unique key based on the shape of the weights
        weight_key = str(weights.shape)
        
        # Initialize velocity for this specific weight matrix if it doesn't exist
        if weight_key not in self.velocity:
            self.velocity[weight_key] = np.zeros_like(weights)
        
        # Update velocity and weights
        self.velocity[weight_key] = self.momentum * self.velocity[weight_key] - self.learning_rate * gradients
        return weights + self.velocity[weight_key]


class NAG(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, **kwargs):
        super().__init__(learning_rate, **kwargs)
        self.momentum = momentum
        self.velocity = {}  # Dictionary to store velocities for each unique weight matrix
    
    def update(self, weights, gradients):
        # Create a unique key based on the shape of the weights
        weight_key = str(weights.shape)
        
        # Initialize velocity for this specific weight matrix if it doesn't exist
        if weight_key not in self.velocity:
            self.velocity[weight_key] = np.zeros_like(weights)
        
        velocity_prev = self.velocity[weight_key].copy()
        
        # Apply weight decay to gradients
        gradients = self.apply_weight_decay(weights, gradients, getattr(self, 'weight_decay', 0))
        
        # Update velocity
        self.velocity[weight_key] = self.momentum * self.velocity[weight_key] - self.learning_rate * gradients
        
        # Apply NAG update
        return weights + self.momentum * velocity_prev + (1 - self.momentum) * self.velocity[weight_key]


class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8, **kwargs):
        super().__init__(learning_rate, **kwargs)
        self.beta = beta
        self.epsilon = epsilon
        self.squared_grad = {}  # Dictionary to store squared gradients for each unique weight matrix
    
    def update(self, weights, gradients):
        gradients = self.apply_weight_decay(weights, gradients, getattr(self, 'weight_decay', 0))
        
        # Create a unique key based on the shape of the weights
        weight_key = str(weights.shape)
        
        # Initialize squared_grad for this specific weight matrix if it doesn't exist
        if weight_key not in self.squared_grad:
            self.squared_grad[weight_key] = np.zeros_like(weights)
        
        self.squared_grad[weight_key] = self.beta * self.squared_grad[weight_key] + (1 - self.beta) * (gradients ** 2)
        return weights - self.learning_rate * gradients / (np.sqrt(self.squared_grad[weight_key]) + self.epsilon)


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        super().__init__(learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # Dictionary to store first moment for each unique weight matrix
        self.v = {}  # Dictionary to store second moment for each unique weight matrix
        self.t = 0
    
    def update(self, weights, gradients):
        gradients = self.apply_weight_decay(weights, gradients, getattr(self, 'weight_decay', 0))
        
        # Create a unique key based on the shape of the weights
        weight_key = str(weights.shape)
        
        # Initialize moments for this specific weight matrix if they don't exist
        if weight_key not in self.m:
            self.m[weight_key] = np.zeros_like(weights)
            self.v[weight_key] = np.zeros_like(weights)
        
        self.t += 1
        
        self.m[weight_key] = self.beta1 * self.m[weight_key] + (1 - self.beta1) * gradients
        self.v[weight_key] = self.beta2 * self.v[weight_key] + (1 - self.beta2) * (gradients ** 2)
        
        m_hat = self.m[weight_key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[weight_key] / (1 - self.beta2 ** self.t)
        
        return weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class NAdam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        super().__init__(learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # Dictionary to store first moment for each unique weight matrix
        self.v = {}  # Dictionary to store second moment for each unique weight matrix
        self.t = 0
    
    def update(self, weights, gradients):
        gradients = self.apply_weight_decay(weights, gradients, getattr(self, 'weight_decay', 0))
        
        # Create a unique key based on the shape of the weights
        weight_key = str(weights.shape)
        
        # Initialize moments for this specific weight matrix if they don't exist
        if weight_key not in self.m:
            self.m[weight_key] = np.zeros_like(weights)
            self.v[weight_key] = np.zeros_like(weights)
        
        self.t += 1
        
        self.m[weight_key] = self.beta1 * self.m[weight_key] + (1 - self.beta1) * gradients
        self.v[weight_key] = self.beta2 * self.v[weight_key] + (1 - self.beta2) * (gradients ** 2)
        
        m_hat = self.m[weight_key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[weight_key] / (1 - self.beta2 ** self.t)
        
        # NAdam update with momentum look-ahead
        m_bar = self.beta1 * m_hat + (1 - self.beta1) * gradients / (1 - self.beta1 ** self.t)
        
        return weights - self.learning_rate * m_bar / (np.sqrt(v_hat) + self.epsilon)


# Neural Network Class
class FeedForwardNN:
    def __init__(self, input_size, hidden_sizes, output_size, activation='sigmoid', weight_init='random'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Set activation function
        if activation == 'identity':
            self.activation_fn = Activation.identity
        elif activation == 'sigmoid':
            self.activation_fn = Activation.sigmoid
        elif activation == 'tanh':
            self.activation_fn = Activation.tanh
        elif activation == 'ReLU':
            self.activation_fn = Activation.relu
        else:
            raise ValueError(f"Activation function {activation} not supported")
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Input layer to first hidden layer
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            if weight_init == 'random':
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            elif weight_init == 'Xavier':
                # Xavier/Glorot initialization for better convergence
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
            else:
                raise ValueError(f"Weight initialization method {weight_init} not supported")
            
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, x):
        """Forward pass through the network"""
        activations = [x]
        pre_activations = []
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(z)
            a = self.activation_fn(z)
            activations.append(a)
        
        # Output layer with softmax for probability distribution
        z_out = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        pre_activations.append(z_out)
        
        # Apply softmax to get probabilities
        exp_scores = np.exp(z_out - np.max(z_out, axis=1, keepdims=True))  # Stabilized softmax
        a_out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        activations.append(a_out)
        
        return activations, pre_activations
    
    def backward(self, x, y, activations, pre_activations, loss_func):
        """Backward pass for computing gradients"""
        batch_size = x.shape[0]
        
        # Initialize gradients
        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer gradient
        delta = loss_func(y, activations[-1], derivative=True)
        
        dw[-1] = np.dot(activations[-2].T, delta) / batch_size
        db[-1] = np.sum(delta, axis=0, keepdims=True) / batch_size
        
        # Backpropagate through hidden layers
        for l in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[l+1].T) * self.activation_fn(pre_activations[l], derivative=True)
            
            dw[l] = np.dot(activations[l].T, delta) / batch_size
            db[l] = np.sum(delta, axis=0, keepdims=True) / batch_size
        
        return dw, db
    
    def update_weights(self, dw, db, optimizer):
        """Update weights using the optimizer"""
        for i in range(len(self.weights)):
            self.weights[i] = optimizer.update(self.weights[i], dw[i])
            self.biases[i] = optimizer.update(self.biases[i], db[i])
    
    def predict(self, x):
        """Predict class probabilities for input x"""
        activations, _ = self.forward(x)
        return activations[-1]
    
    def accuracy(self, x, y):
        """Calculate accuracy"""
        predictions = self.predict(x)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        return np.mean(predicted_classes == true_classes)


def preprocess_data(x_train, y_train, x_test, y_test):
    """Preprocess data"""
    # Reshape and normalize images
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
    
    # Convert labels to one-hot encoding
    def one_hot_encode(y, num_classes=10):
        return np.eye(num_classes)[y]
    
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)
    
    return x_train, y_train, x_test, y_test


def get_batches(x, y, batch_size):
    """Generate batches for training"""
    indices = np.random.permutation(len(x))
    for i in range(0, len(x), batch_size):
        batch_idx = indices[i:i + batch_size]
        yield x[batch_idx], y[batch_idx]


def get_optimizer(args):
    """Initialize optimizer based on arguments"""
    optimizer_params = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay
    }
    
    if args.optimizer == 'sgd':
        return SGD(**optimizer_params)
    elif args.optimizer == 'momentum':
        return Momentum(momentum=args.momentum, **optimizer_params)
    elif args.optimizer == 'nag':
        return NAG(momentum=args.momentum, **optimizer_params)
    elif args.optimizer == 'rmsprop':
        return RMSProp(beta=args.beta, epsilon=args.epsilon, **optimizer_params)
    elif args.optimizer == 'adam':
        return Adam(beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon, **optimizer_params)
    elif args.optimizer == 'nadam':
        return NAdam(beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon, **optimizer_params)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")


def get_loss_function(loss_name):
    """Get loss function based on name"""
    if loss_name == 'mean_squared_error':
        return Loss.mean_squared_error
    elif loss_name == 'cross_entropy':
        return Loss.cross_entropy
    else:
        raise ValueError(f"Loss function {loss_name} not supported")


def train(args):
    """Train the neural network"""
    # Initialize wandb
    wandb.login(key='297e38b27732dc76bf218c8d24bbc10053006b3e')
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    wandb.config.update(args)
    
    # Load dataset
    if args.dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    # Preprocess data
    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)
    
    # Split training data into training and validation sets (90-10)
    val_size = int(0.1 * len(x_train))
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]
    
    # Initialize the neural network
    input_size = x_train.shape[1]  # 784 for Fashion-MNIST
    hidden_sizes = [args.hidden_size] * args.num_layers
    
    model = FeedForwardNN(input_size, hidden_sizes, 10, args.activation, args.weight_init)
    
    # Get loss function
    loss_func = get_loss_function(args.loss)
    
    # Initialize optimizer
    optimizer = get_optimizer(args)
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        total_loss = 0
        batch_count = 0
        
        # Training phase
        for batch_x, batch_y in tqdm(get_batches(x_train, y_train, args.batch_size),
                                     total=len(x_train)//args.batch_size,
                                     desc=f"Epoch {epoch+1}/{args.epochs}"):
            # Forward pass
            activations, pre_activations = model.forward(batch_x)
            
            # Compute loss
            loss = loss_func(batch_y, activations[-1])
            total_loss += loss
            batch_count += 1
            
            # Backward pass
            dw, db = model.backward(batch_x, batch_y, activations, pre_activations, loss_func)
            
            # Update weights
            model.update_weights(dw, db, optimizer)
        
        # Calculate metrics
        train_loss = total_loss / batch_count
        train_accuracy = model.accuracy(x_train, y_train)
        val_accuracy = model.accuracy(x_val, y_val)
        test_accuracy = model.accuracy(x_test, y_test)
        
        # Log metrics
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy
        })
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Loss: {train_loss:.4f} - "
              f"Train Acc: {train_accuracy:.4f} - "
              f"Val Acc: {val_accuracy:.4f} - "
              f"Test Acc: {test_accuracy:.4f}")
    
    print("Training completed!")
    
    # Final evaluation
    final_test_accuracy = model.accuracy(x_test, y_test)
    print(f"Final Test Accuracy: {final_test_accuracy:.4f}")
    
    wandb.finish()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a feedforward neural network on Fashion-MNIST")
    
    # WandB arguments
    parser.add_argument('-wp', '--wandb_project', type=str, default='DA640_A01',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', type=str, default='prahaladvathsan-iit-madras',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    
    # Dataset
    parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion_mnist'],
                        default='fashion_mnist', help='Dataset to use')
    
    # Training parameters
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs to train neural network')
    parser.add_argument('-b', '--batch_size', type=int, default=4,
                        help='Batch size used to train neural network')
    
    # Loss function
    parser.add_argument('-l', '--loss', type=str, choices=['mean_squared_error', 'cross_entropy'],
                        default='cross_entropy', help='Loss function')
    
    # Optimizer
    parser.add_argument('-o', '--optimizer', type=str,
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        default='sgd', help='Optimizer')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,
                        help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum', type=float, default=0.5,
                        help='Momentum used by momentum and nag optimizers')
    parser.add_argument('-beta', '--beta', type=float, default=0.5,
                        help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5,
                        help='Beta1 used by adam and nadam optimizers')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5,
                        help='Beta2 used by adam and nadam optimizers')
    parser.add_argument('-eps', '--epsilon', type=float, default=0.000001,
                        help='Epsilon used by optimizers')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0,
                        help='Weight decay used by optimizers')
    
    # Network configuration
    parser.add_argument('-w_i', '--weight_init', type=str, choices=['random', 'Xavier'],
                        default='random', help='Weight initialization method')
    parser.add_argument('-nhl', '--num_layers', type=int, default=1,
                        help='Number of hidden layers used in feedforward neural network')
    parser.add_argument('-sz', '--hidden_size', type=int, default=4,
                        help='Number of hidden neurons in a feedforward layer')
    parser.add_argument('-a', '--activation', type=str, choices=['identity', 'sigmoid', 'tanh', 'ReLU'],
                        default='sigmoid', help='Activation function')
    
    args = parser.parse_args()
    
    # Train the model
    train(args)