import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.datasets import fashion_mnist
import argparse
import wandb

# Import your neural network implementation
from Neural_Network import FeedForwardNN, preprocess_data

def evaluate_best_model():
    """Evaluate the best model and create confusion matrix visualization"""
    parser = argparse.ArgumentParser(description="Evaluate best model from hyperparameter sweep")
    parser.add_argument('-wp', '--wandb_project', type=str, default='DA640_A01',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', type=str, default='prahaladvathsan-iit-madras',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    
    # Add hyperparameters for your best model (replace with your best values)
    parser.add_argument('-nhl', '--num_layers', type=int, default=3,
                        help='Number of hidden layers in the best model')
    parser.add_argument('-sz', '--hidden_size', type=int, default=128,
                        help='Size of hidden layers in the best model')
    parser.add_argument('-a', '--activation', type=str, default='ReLU',
                        help='Activation function used in the best model')
    parser.add_argument('-w_i', '--weight_init', type=str, default='Xavier',
                        help='Weight initialization used in the best model')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of epochs to train the best model')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='Batch size used to train the best model')
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        help='Optimizer used in the best model')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Learning rate used in the best model')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0005,
                        help='Weight decay used in the best model')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)
    
    # Class names for Fashion-MNIST
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Initialize wandb (optional, if you want to log the confusion matrix)
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name="best_model_evaluation")
    
    # Initialize and train the best model
    print("Initializing the best model...")
    hidden_sizes = [args.hidden_size] * args.num_layers
    
    best_model = FeedForwardNN(
        input_size=x_train.shape[1],
        hidden_sizes=hidden_sizes,
        output_size=10,
        activation=args.activation,
        weight_init=args.weight_init
    )
    
    # Here you would train the model with the best hyperparameters
    # Or load a pre-trained model if you saved it
    
    # For this example, we'll just evaluate on the test set
    print("Evaluating the model on test set...")
    predictions = best_model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate test accuracy
    test_accuracy = np.mean(predicted_classes == true_classes)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create a more creative visualization
    plt.figure(figsize=(12, 10))
    
    # Use a custom colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Plot the confusion matrix with a heatmap
    ax = sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=.5, cbar_kws={"shrink": .8})
    
    # Improve the visualization
    plt.title('Fashion-MNIST Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            # Calculate percentage
            pct = cm_normalized[i, j] * 100
            # Add percentage text in each cell
            if pct > 50:
                color = 'white'
            else:
                color = 'black'
            # Add percentage below the count
            ax.text(j+0.5, i+0.7, f"{pct:.1f}%", 
                    ha="center", va="center", color=color, fontsize=9)
    
    # Add a descriptive subtitle
    plt.figtext(0.5, 0.01, 
                f"Best Model: {args.num_layers} layers × {args.hidden_size} neurons, {args.activation} activation, {args.optimizer} optimizer", 
                ha="center", fontsize=12)
    
    # Highlight diagonal elements (correct predictions)
    for i in range(len(class_names)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', lw=2))
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('fashion_mnist_confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    # Log the confusion matrix to wandb (optional)
    wandb.log({"confusion_matrix": wandb.Image('fashion_mnist_confusion_matrix.png'),
              "test_accuracy": test_accuracy})
    
    print("Confusion matrix saved to 'fashion_mnist_confusion_matrix.png'")
    
    # Create additional visualizations for misclassified examples
    # Find indices of misclassified samples
    misclassified_indices = np.where(predicted_classes != true_classes)[0]
    
    if len(misclassified_indices) > 0:
        # Plot some misclassified examples
        plt.figure(figsize=(15, 10))
        
        # Select up to 25 misclassified examples to show
        num_examples = min(25, len(misclassified_indices))
        selected_indices = np.random.choice(misclassified_indices, num_examples, replace=False)
        
        for i, idx in enumerate(selected_indices):
            plt.subplot(5, 5, i + 1)
            
            # Reshape the flattened image back to 28x28
            img = x_test[idx].reshape(28, 28)
            
            # Display the image
            plt.imshow(img, cmap='gray')
            
            # Add title with true and predicted labels
            true_label = class_names[true_classes[idx]]
            pred_label = class_names[predicted_classes[idx]]
            plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=8)
            
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('misclassified_examples.png', dpi=300, bbox_inches='tight')
        
        # Log to wandb (optional)
        wandb.log({"misclassified_examples": wandb.Image('misclassified_examples.png')})
        
        print("Misclassified examples saved to 'misclassified_examples.png'")
    
    wandb.finish()

if __name__ == "__main__":
    evaluate_best_model()