import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Load the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Class names for Fashion-MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Create a 2x5 grid of images
plt.figure(figsize=(12, 6))

# Plot one sample for each class
for i in range(10):
    # Find the first occurrence of class i
    idx = np.where(y_train == i)[0][0]
    
    # Create subplot
    plt.subplot(2, 5, i + 1)
    
    # Display the image
    plt.imshow(x_train[idx], cmap='gray')
    
    # Add title with class name
    plt.title(f"{i}: {class_names[i]}")
    
    # Turn off axis
    plt.axis('off')

plt.tight_layout()
plt.savefig('fashion_mnist_samples.png')
plt.show()