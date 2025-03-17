import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from keras.datasets import fashion_mnist
import wandb

# Import your neural network implementation
from Neural_Network import FeedForwardNN, preprocess_data, get_optimizer

def create_enhanced_confusion_matrix(model, x_test, y_test, class_names):
    """
    Creates an enhanced confusion matrix visualization with additional metrics
    
    Parameters:
    -----------
    model : FeedForwardNN
        The trained neural network model
    x_test : numpy.ndarray
        Test data
    y_test : numpy.ndarray
        Test labels (one-hot encoded)
    class_names : list
        List of class names
    """
    # Get predictions
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate test accuracy
    test_accuracy = np.mean(predicted_classes == true_classes)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Calculate precision, recall, and F1 for each class
    precision, recall, f1, _ = precision_recall_fscore_support(true_classes, predicted_classes, average=None)
    
    # Create a comprehensive figure with multiple components
    plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
    
    # Plot the main confusion matrix
    ax0 = plt.subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=.5, cbar=False, ax=ax0)
    
    ax0.set_title(f'Fashion-MNIST Confusion Matrix\nTest Accuracy: {test_accuracy:.4f}', fontsize=16)
    ax0.set_ylabel('True Label', fontsize=12)
    ax0.set_xlabel('Predicted Label', fontsize=12)
    
    # Rotate labels for better visibility
    plt.setp(ax0.get_xticklabels(), rotation=45, ha='right')
    
    # Plot precision, recall, and F1 scores
    ax1 = plt.subplot(gs[0, 1])
    metrics_df = np.column_stack((precision, recall, f1))
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='Greens',
                xticklabels=['Precision', 'Recall', 'F1'],
                yticklabels=class_names, linewidths=.5, ax=ax1)
    ax1.set_title('Class-wise Metrics', fontsize=14)
    
    # Plot class distribution
    ax2 = plt.subplot(gs[1, 0])
    class_counts = np.bincount(true_classes)
    incorrect_counts = np.array([np.sum(
        (true_classes == i) & (predicted_classes != i)) for i in range(len(class_names))])
    
    # Create stacked bars showing correct and incorrect counts
    bar_width = 0.8
    x = np.arange(len(class_names))
    
    ax2.bar(x, class_counts - incorrect_counts, bar_width, label='Correct', color='#4CAF50')
    ax2.bar(x, incorrect_counts, bar_width, bottom=class_counts - incorrect_counts, 
           label='Incorrect', color='#F44336')
    
    ax2.set_xlabel('Classes', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Class Distribution and Accuracy', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    
    # Plot confusion patterns (most common misclassifications)
    ax3 = plt.subplot(gs[1, 1])
    
    # Get the most commonly confused pairs (excluding diagonal)
    confusion_flat = cm.copy()
    np.fill_diagonal(confusion_flat, 0)  # Zero out diagonal
    most_confused = np.unravel_index(np.argsort(confusion_flat.ravel())[-5:], cm.shape)
    confused_pairs = list(zip(most_confused[0], most_confused[1]))
    confused_pairs.reverse()  # Show highest first
    
    confusions = [f"{class_names[true]} → {class_names[pred]}: {cm[true, pred]}" 
                 for true, pred in confused_pairs]
    
    # Plot as text
    ax3.axis('off')
    ax3.set_title('Top 5 Confusion Patterns', fontsize=14)
    y_pos = 0.8
    for conf in confusions:
        ax3.text(0.5, y_pos, conf, ha='center', fontsize=12)
        y_pos -= 0.15
    
    plt.tight_layout()
    plt.savefig('enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    return test_accuracy, cm, precision, recall, f1

def train_and_evaluate_best_model():
    """
    Train and evaluate the best model from hyperparameter search
    """
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)
    
    # Class names for Fashion-MNIST
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Initialize wandb (optional)
    wandb.init(project="DA640_A01", name="best_model_analysis")
    
    # Define best hyperparameters (update these with your best values from sweeps)
    num_layers = 4
    hidden_size = 128
    activation = 'ReLU'
    weight_init = 'Xavier'
    optimizer_name = 'adam'
    learning_rate = 0.001
    weight_decay = 0.0005
    batch_size = 32
    epochs = 10
    
    # Initialize the best model
    print("Initializing the best model...")
    hidden_sizes = [hidden_size] * num_layers
    
    best_model = FeedForwardNN(
        input_size=x_train.shape[1],
        hidden_sizes=hidden_sizes,
        output_size=10,
        activation=activation,
        weight_init=weight_init
    )
    
    # Initialize optimizer with best parameters
    from types import SimpleNamespace
    opt_args = SimpleNamespace(
        optimizer=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=0.9,
        beta=0.9,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    )
    
    optimizer = get_optimizer(opt_args)
    
    # Train the model with best hyperparameters
    print(f"Training the best model ({num_layers} layers, {hidden_size} neurons, {activation}, {optimizer_name})...")
    
    # Here you would add the training code similar to what you have in your main training function
    # For brevity, I'm omitting the actual training loop
    
    # After training, evaluate the model
    print("Evaluating the model and creating enhanced visualization...")
    test_accuracy, cm, precision, recall, f1 = create_enhanced_confusion_matrix(
        best_model, x_test, y_test, class_names)
    
    # Log to wandb (optional)
    wandb.log({
        "test_accuracy": test_accuracy,
        "confusion_matrix": wandb.Image('enhanced_confusion_matrix.png'),
        "precision": precision.mean(),
        "recall": recall.mean(),
        "f1": f1.mean()
    })
    
    # Create a visualization of misclassified examples
    print("Generating visualization of misclassified examples...")
    
    # Get predictions
    predictions = best_model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Find indices of misclassified samples
    misclassified_indices = np.where(predicted_classes != true_classes)[0]
    
    if len(misclassified_indices) > 0:
        # Plot some examples from the most confused classes
        plt.figure(figsize=(15, 12))
        
        # Get top confusion pairs
        confusion_flat = cm.copy()
        np.fill_diagonal(confusion_flat, 0)  # Zero out diagonal
        most_confused = np.unravel_index(np.argsort(confusion_flat.ravel())[-5:], cm.shape)
        confused_pairs = list(zip(most_confused[0], most_confused[1]))
        
        # Plot examples for each confusion pair
        pair_idx = 1
        for true_class, pred_class in confused_pairs:
            # Find examples of this confusion
            confusion_indices = np.where((true_classes == true_class) & 
                                         (predicted_classes == pred_class))[0]
            
            if len(confusion_indices) > 0:
                # Show up to 5 examples of this confusion
                num_examples = min(5, len(confusion_indices))
                for i in range(num_examples):
                    idx = confusion_indices[i]
                    plt.subplot(5, 5, pair_idx)
                    
                    # Reshape the flattened image back to 28x28
                    img = x_test[idx].reshape(28, 28)
                    
                    # Display the image
                    plt.imshow(img, cmap='gray')
                    
                    # Add title with true and predicted labels
                    plt.title(f"True: {class_names[true_class]}\nPred: {class_names[pred_class]}", fontsize=8)
                    plt.axis('off')
                    
                    pair_idx += 1
                    if pair_idx > 25:
                        break
                
            if pair_idx > 25:
                break
        
        plt.tight_layout()
        plt.savefig('common_confusions.png', dpi=300, bbox_inches='tight')
        
        # Log to wandb (optional)
        wandb.log({"common_confusions": wandb.Image('common_confusions.png')})
    
    wandb.finish()
    print("Analysis complete!")

if __name__ == "__main__":
    train_and_evaluate_best_model()