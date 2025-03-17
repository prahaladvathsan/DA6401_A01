import numpy as np
import wandb
import argparse
from keras.datasets import fashion_mnist
from tqdm import tqdm
import os
import sys
from types import SimpleNamespace

# Import the neural network implementation from the main file
from Neural_Network import FeedForwardNN, get_optimizer, get_loss_function, preprocess_data, get_batches

def train_model_for_sweep():
    """Training function used by wandb sweep agent"""
    # Initialize wandb run with config values
    with wandb.init() as run:
        config = wandb.config
        
        # Load fashion_mnist dataset
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        
        # Preprocess data
        x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)
        
        # Split training data into train and validation (90-10)
        val_size = int(0.1 * len(x_train))
        x_val, y_val = x_train[-val_size:], y_train[-val_size:]
        x_train, y_train = x_train[:-val_size], y_train[:-val_size]
        
        # Set up hidden layer sizes (all equal size in this implementation)
        hidden_sizes = [config.hidden_size] * config.num_layers
        
        # Initialize the neural network
        model = FeedForwardNN(
            input_size=x_train.shape[1],  # 784 for Fashion-MNIST
            hidden_sizes=hidden_sizes,
            output_size=10,
            activation=config.activation,
            weight_init=config.weight_init
        )
        
        # Get loss function
        loss_func = get_loss_function(config.loss)
        
        # Initialize optimizer with config parameters
        # Convert config dictionary to an object with attributes
        opt_args = SimpleNamespace(
            optimizer=config.optimizer,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum if hasattr(config, 'momentum') else 0.9,
            beta=config.beta if hasattr(config, 'beta') else 0.9,
            beta1=config.beta1 if hasattr(config, 'beta1') else 0.9,
            beta2=config.beta2 if hasattr(config, 'beta2') else 0.999,
            epsilon=config.epsilon if hasattr(config, 'epsilon') else 1e-8
        )
        
        optimizer = get_optimizer(opt_args)
        
        # Training loop
        for epoch in range(config.epochs):
            total_loss = 0
            batch_count = 0
            
            # Training phase
            for batch_x, batch_y in get_batches(x_train, y_train, config.batch_size):
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
            
            # Log metrics to wandb
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'test_accuracy': test_accuracy
            })
        
        # Log final metrics
        final_val_accuracy = model.accuracy(x_val, y_val)
        final_test_accuracy = model.accuracy(x_test, y_test)
        
        # Log summary metrics for easy comparison in wandb
        wandb.run.summary["final_val_accuracy"] = final_val_accuracy
        wandb.run.summary["final_test_accuracy"] = final_test_accuracy
        
        # Create meaningful run name based on hyperparameters
        run_name = f"hl_{config.num_layers}_bs_{config.batch_size}_act_{config.activation}_opt_{config.optimizer}"
        wandb.run.name = run_name
        wandb.run.save()
        
        return final_val_accuracy

# Define sweep configuration
sweep_config = {
    'method': 'bayes',  # Bayesian optimization method
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'values': [5, 10]
        },
        'num_layers': {
            'values': [3, 4, 5]
        },
        'hidden_size': {
            'values': [32, 64, 128]
        },
        'weight_decay': {
            'values': [0, 0.0005, 0.5]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'optimizer': {
            'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'weight_init': {
            'values': ['random', 'Xavier']
        },
        'activation': {
            'values': ['sigmoid', 'tanh', 'ReLU']
        },
        # Default values for other parameters
        'loss': {'value': 'cross_entropy'},
        'momentum': {'value': 0.9},
        'beta': {'value': 0.9},
        'beta1': {'value': 0.9},
        'beta2': {'value': 0.999},
        'epsilon': {'value': 1e-8}
    }
}

def main():
    parser = argparse.ArgumentParser(description="Run WandB hyperparameter sweep")
    parser.add_argument('-wp', '--wandb_project', type=str, default='DA640_A01',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', type=str, default='prahaladvathsan-iit-madras',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    parser.add_argument('--sweep_count', type=int, default=100,
                       help='Number of runs to execute in the sweep')
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.login()
    
    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
    
    # Start the sweep agent
    wandb.agent(sweep_id, train_model_for_sweep, count=args.sweep_count)

if __name__ == "__main__":
    main()