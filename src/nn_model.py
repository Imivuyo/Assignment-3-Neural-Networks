import numpy as np
import tensorflow as tf

class FeedforwardNN:
    """Single hidden layer feedforward neural network with ReLU and softmax/linear output"""
    
    def __init__(self, input_dim, hidden_units, output_dim, problem_type):
        """Initialize architecture with He initialization"""
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.problem_type = problem_type
        
        # He initialization for ReLU
        he_std = np.sqrt(2 / input_dim)
        self.W1 = np.random.normal(0, he_std, (input_dim, hidden_units))
        self.b1 = np.zeros((hidden_units,))
        he_std = np.sqrt(2 / hidden_units)
        self.W2 = np.random.normal(0, he_std, (hidden_units, output_dim))
        self.b2 = np.zeros((output_dim,))
        
        # Define loss function
        if problem_type == 'classification':
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        else:
            self.loss_fn = tf.keras.losses.Huber(delta=1.0)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X, training=True):
        """Forward pass"""
        self.X = X
        self.z1 = np.dot(X, self.W1) + self.b1
        self.h1 = self.relu(self.z1)
        self.z2 = np.dot(self.h1, self.W2) + self.b2
        if self.problem_type == 'classification':
            self.y_pred = self.softmax(self.z2)
        else:
            self.y_pred = self.z2
        return self.y_pred
    
    def compute_loss(self, y_true, y_pred):
        """Compute loss (cross-entropy for classification, Huber for regression)"""
        if self.problem_type == 'regression':
            y_true = y_true.reshape(-1, self.output_dim)
        return self.loss_fn(y_true, y_pred).numpy()
    
    def backward(self, y_true):
        """Compute gradients for backpropagation"""
        if self.problem_type == 'classification':
            grad_z2 = self.y_pred - y_true
        else:
            y_true = y_true.reshape(-1, self.output_dim)
            delta = 1.0
            error = self.y_pred - y_true
            huber_mask = np.abs(error) <= delta
            grad_z2 = np.where(huber_mask, error, delta * np.sign(error))
        
        grad_W2 = np.dot(self.h1.T, grad_z2) / y_true.shape[0]
        grad_b2 = np.mean(grad_z2, axis=0)
        
        grad_h1 = np.dot(grad_z2, self.W2.T)
        grad_z1 = grad_h1 * (self.z1 > 0)
        grad_W1 = np.dot(self.X.T, grad_z1) / y_true.shape[0]
        grad_b1 = np.mean(grad_z1, axis=0)
        
        return {'W1': grad_W1, 'b1': grad_b1, 'W2': grad_W2, 'b2': grad_b2}
    
    def get_weights_flat(self):
        """Return flattened weights for SCG and LeapFrog trainers"""
        return np.concatenate([
            self.W1.flatten(), self.b1.flatten(),
            self.W2.flatten(), self.b2.flatten()
        ])
    
    def set_weights_flat(self, flat_weights):
        """Set weights from flattened array"""
        idx = 0
        w1_size = self.input_dim * self.hidden_units
        self.W1 = flat_weights[idx:idx + w1_size].reshape(self.input_dim, self.hidden_units)
        idx += w1_size
        b1_size = self.hidden_units
        self.b1 = flat_weights[idx:idx + b1_size]
        idx += b1_size
        w2_size = self.hidden_units * self.output_dim
        self.W2 = flat_weights[idx:idx + w2_size].reshape(self.hidden_units, self.output_dim)
        idx += w2_size
        self.b2 = flat_weights[idx:]

    def predict(self, X):
        """Predict on new data"""
        y_pred = self.forward(X, training=False)
        if self.problem_type == 'classification':
            return np.argmax(y_pred, axis=1)
        return y_pred.flatten()