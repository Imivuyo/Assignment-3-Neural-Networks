import numpy as np
import time
import random
import tensorflow as tf

from config import CONFIG

class SGDTrainer:
    """Stochastic Gradient Descent with momentum, mini-batches, and early stopping"""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def train(self, model, X_train, y_train, X_val, y_val, max_epochs=CONFIG['max_epochs'], patience=CONFIG['early_stopping_patience'], batch_size=CONFIG['batch_size']):
        """Train the model using mini-batch SGD"""
        history = {
            'train_loss_epochs': [],
            'val_loss_epochs': [],
            'epochs': 0
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = model.get_weights_flat().copy()
        n_samples = X_train.shape[0]
        
        if self.velocity is None:
            self.velocity = {
                'W1': np.zeros_like(model.W1),
                'b1': np.zeros_like(model.b1),
                'W2': np.zeros_like(model.W2),
                'b2': np.zeros_like(model.b2)
            }
        
        for epoch in range(max_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            train_loss = 0
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward and loss
                y_pred = model.forward(X_batch)
                batch_loss = model.compute_loss(y_batch, y_pred)
                train_loss += batch_loss * (end - start)
                
                # Backward
                gradients = model.backward(y_batch)
                
                # Update with momentum
                for key in gradients:
                    self.velocity[key] = (self.momentum * self.velocity[key] - 
                                         self.learning_rate * gradients[key])
                    setattr(model, key, getattr(model, key) + self.velocity[key])
            
            train_loss /= n_samples
            history['train_loss_epochs'].append(train_loss)
            
            # Validation
            y_val_pred = model.forward(X_val)
            val_loss = model.compute_loss(y_val, y_val_pred)
            history['val_loss_epochs'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.get_weights_flat().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                history['epochs'] = epoch + 1
                break
        
        model.set_weights_flat(best_weights)
        if patience_counter < patience:
            history['epochs'] = max_epochs
        
        return history

class SCGTrainer:
    """Scaled Conjugate Gradient (Møller, 1993)"""
    
    def __init__(self, sigma=1e-4, lambda_=1e-6):
        self.sigma = sigma
        self.lambda_ = lambda_
    
    def train(self, model, X_train, y_train, X_val, y_val, max_epochs=CONFIG['max_epochs'], patience=CONFIG['early_stopping_patience']):
        """Train using SCG algorithm following Moller 1993"""
        history = {
            'train_loss_epochs': [],
            'val_loss_epochs': [],
            'epochs': 0
        }
        
        w = model.get_weights_flat()
        n_weights = len(w)
        r = -self._get_gradient(model, X_train, y_train)
        p = r.copy()
        lambda_k = self.lambda_
        lambda_bar_k = 0.0
        success = True
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_w = w.copy()
        
        k = 1
        
        for epoch in range(max_epochs):
            if success:
                sigma_k = self.sigma / (np.linalg.norm(p) + 1e-12)
                w_plus = w + sigma_k * p
                model.set_weights_flat(w_plus)
                r_plus = -self._get_gradient(model, X_train, y_train)
                model.set_weights_flat(w)
                s_k = (r_plus - r) / sigma_k
                delta_k = np.dot(p, s_k)
            
            delta_k = delta_k + (lambda_k - lambda_bar_k) * np.dot(p, p)
            
            if delta_k <= 0:
                lambda_bar_k = 2 * (lambda_k - delta_k / (np.dot(p, p) + 1e-12))
                delta_k = -delta_k + lambda_k * np.dot(p, p)
                lambda_k = lambda_bar_k
            
            mu_k = np.dot(p, r)
            if abs(delta_k) < 1e-12:
                delta_k = 1e-12
            alpha_k = mu_k / delta_k
            
            w_new = w + alpha_k * p
            model.set_weights_flat(w_new)
            E_new = self._get_loss(model, X_train, y_train)
            model.set_weights_flat(w)
            E_old = self._get_loss(model, X_train, y_train)
            
            mu_k_safe = mu_k if abs(mu_k) >= 1e-12 else 1e-12
            Delta_k = 2 * delta_k * (E_old - E_new) / (mu_k_safe ** 2)
            
            if Delta_k >= 0:
                w = w_new
                model.set_weights_flat(w)
                r_new = -self._get_gradient(model, X_train, y_train)
                lambda_bar_k = 0
                success = True
                
                if np.linalg.norm(r_new) < 1e-8:
                    history['train_loss_epochs'].append(E_new)
                    val_loss = self._get_loss(model, X_val, y_val)
                    history['val_loss_epochs'].append(val_loss)
                    history['epochs'] = epoch + 1
                    break
                
                if k % n_weights == 0:
                    p = r_new.copy()
                else:
                    beta_k = (np.dot(r_new, r_new) - np.dot(r_new, r)) / (mu_k_safe + 1e-12)
                    p = r_new + beta_k * p
                
                if Delta_k >= 0.75:
                    lambda_k = 0.25 * lambda_k
                
                r = r_new
            else:
                lambda_bar_k = lambda_k
                success = False
            
            if Delta_k < 0.25:
                lambda_k = lambda_k + (delta_k * (1 - Delta_k) / (np.dot(p, p) + 1e-12))
            
            if success or epoch == 0:
                current_weights = w if success else best_w
                model.set_weights_flat(current_weights)
                train_loss = self._get_loss(model, X_train, y_train)
                val_loss = self._get_loss(model, X_val, y_val)
                history['train_loss_epochs'].append(train_loss)
                history['val_loss_epochs'].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_w = current_weights.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    history['epochs'] = epoch + 1
                    break
            
            k += 1
        
        model.set_weights_flat(best_w)
        if patience_counter < patience and epoch == max_epochs - 1:
            history['epochs'] = max_epochs
        
        return history
    
    def _get_gradient(self, model, X, y):
        model.forward(X)
        grads = model.backward(y)
        return np.concatenate([grads['W1'].flatten(), grads['b1'].flatten(), 
                              grads['W2'].flatten(), grads['b2'].flatten()])
    
    def _get_loss(self, model, X, y):
        y_pred = model.forward(X)
        return model.compute_loss(y, y_pred)

class LeapFrogTrainer:
    """LeapFrog Optimization (Snyman LFOP1(b), 1983)"""
    
    def __init__(self, delta_t=0.1, delta=0.5, m=3, delta_1=0.001, j=2, M=10, N_max=2, epsilon=1e-5):
        self.delta_t = delta_t
        self.delta = delta
        self.m = m
        self.delta_1 = delta_1
        self.j = j
        self.M = M
        self.N_max = N_max
        self.epsilon = epsilon
    
    def train(self, model, X_train, y_train, X_val, y_val, max_epochs=CONFIG['max_epochs'], patience=CONFIG['early_stopping_patience']):
        """Train using LFOP1(b) algorithm"""
        history = {'train_loss_epochs': [], 'val_loss_epochs': [], 'epochs': 0}
        
        w = model.get_weights_flat()
        model.set_weights_flat(w)
        a_prev = -self._get_gradient(model, X_train, y_train)
        v = 0.5 * a_prev * self.delta_t
        
        i = 0
        s = 0
        N = 1
        consecutive_max_steps = 0
        time_step_reductions = 0
        best_val_loss = float('inf')
        patience_counter = 0
        best_w = w.copy()
        
        for epoch in range(max_epochs):
            w_prev = w.copy()
            v_prev = v.copy()
            a_prev_prev = a_prev.copy()
            
            v_norm = np.linalg.norm(v)
            delta_x_norm = v_norm * self.delta_t
            
            step_at_max_size = False
            if delta_x_norm >= self.delta and v_norm > 0:
                v = (self.delta * v) / (self.delta_t * v_norm)
                step_at_max_size = True
                consecutive_max_steps += 1
            else:
                consecutive_max_steps = 0
            
            if consecutive_max_steps >= self.M and time_step_reductions < self.N_max:
                self.delta_t /= 4.0
                w = (w + w_prev) / 2
                v_scaled = (self.delta * v) / (self.delta_t * np.linalg.norm(v)) if np.linalg.norm(v) > 0 else v
                v = (v_scaled + v_prev) / 4
                consecutive_max_steps = 0
                time_step_reductions += 1
                model.set_weights_flat(w)
                a_prev = -self._get_gradient(model, X_train, y_train)
                continue
            
            w_new = w + v * self.delta_t
            model.set_weights_flat(w_new)
            a_new = -self._get_gradient(model, X_train, y_train)
            v_new = v + a_new * self.delta_t
            
            grad_aligned = np.dot(a_new, a_prev) > 0
            step_within_bounds = delta_x_norm < self.delta
            
            if grad_aligned and step_within_bounds:
                growth_factor = 1 + N * self.delta_1
                max_growth = 2.0
                actual_growth = min(growth_factor, max_growth)
                self.delta_t *= actual_growth
                N += 1
                s = 0
            else:
                if not grad_aligned:
                    s += 1
                N = 1
            
            if s >= self.m:
                self.delta_t /= 2.0
                w_new = (w_new + w_prev) / 2
                v_new = (v_new + v_prev) / 4
                s = 0
                model.set_weights_flat(w_new)
                a_new = -self._get_gradient(model, X_train, y_train)
            
            if np.linalg.norm(a_new) <= self.epsilon:
                w, v, a_prev = w_new, v_new, a_new
                model.set_weights_flat(w)
                break
            
            v_new_norm = np.linalg.norm(v_new)
            v_norm_prev = np.linalg.norm(v)
            
            if v_new_norm > v_norm_prev:
                i = 0
                w, v, a_prev = w_new, v_new, a_new
            else:
                i += 1
                w_interfered = (w_new + w_prev) / 2
                if i <= self.j:
                    v_interfered = (v_new + v_prev) / 4
                    w, v = w_interfered, v_interfered
                    model.set_weights_flat(w)
                    a_prev = -self._get_gradient(model, X_train, y_train)
                else:
                    w = w_interfered
                    v = np.zeros_like(v)
                    i = 0
                    model.set_weights_flat(w)
                    a_prev = -self._get_gradient(model, X_train, y_train)
            
            current_weights = w.copy()
            model.set_weights_flat(current_weights)
            train_loss = self._get_loss(model, X_train, y_train)
            val_loss = self._get_loss(model, X_val, y_val)
            history['train_loss_epochs'].append(train_loss)
            history['val_loss_epochs'].append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_w = current_weights.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    history['epochs'] = epoch + 1
                    break
        
        model.set_weights_flat(best_w)
        if patience_counter < patience and history['epochs'] == 0:
            history['epochs'] = max_epochs
        
        return history
    
    def _get_gradient(self, model, X, y):
        model.forward(X)
        grads = model.backward(y)
        return np.concatenate([
            grads['W1'].flatten(), grads['b1'].flatten(),
            grads['W2'].flatten(), grads['b2'].flatten()
        ])
    
    def _get_loss(self, model, X, y):
        y_pred = model.forward(X)
        return model.compute_loss(y, y_pred)