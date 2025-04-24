from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.values                  # Convert from pandas to NumPy
y = y.astype(int).values      # Convert labels to integer

# Print dataset shapes
print(X.shape)  # (70000, 784)
print(y.shape)  # (70000,)

# Normalize the pixel values to [-1, 1]
X = ((X / 255.) - 0.5) * 2
print(X)

# Visualize one image per digit from 0 to 9
for i in range(10):
    img = X[y == i][0].reshape(28, 28)
    # plt.imshow(img, cmap='Greys')
    # plt.show()

# Visualize 25 samples of digit '7'
for i in range(25):
    img = X[y == 7][i].reshape(28, 28)
    # plt.imshow(img, cmap='Greys')
    # plt.show()

# Split the dataset: 10k test, 5k validation, rest training
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=10000, random_state=123, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)

# --- MLP implementation starts here ---

# Sigmoid activation function
def sigmoid(z):
    return 1. / (1. + np.exp(-z))

# Convert class labels to one-hot encoded vectors
def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary

# Multilayer Perceptron class
class NeuralNetMLP:

    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()
        self.num_classes = num_classes
        rng = np.random.RandomState(random_seed)

        # Init weights and biases for hidden layer
        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        # Init weights and biases for output layer
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, X):
        # Compute activations for hidden layer
        z_h = np.dot(X, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # Compute activations for output layer
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)

        return a_h, a_out

    def backward(self, X, a_h, a_out, y):
        # One-hot encode the labels
        y_onehot = int_to_onehot(y, self.num_classes)

        # Compute gradient of loss w.r.t. output
        d_loss__d_a_out = 2. * (a_out - y_onehot) / y.shape[0]
        d_a_out__d_z_out = a_out * (1. - a_out)
        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        # Gradients for output weights and biases
        d_loss__dw_out = np.dot(delta_out.T, a_h)
        d_loss__db_out = np.sum(delta_out, axis=0)

        # Backprop into hidden layer
        d_loss__a_h = np.dot(delta_out, self.weight_out)
        d_a_h__d_z_h = a_h * (1. - a_h)
        delta_h = d_loss__a_h * d_a_h__d_z_h

        # Gradients for hidden weights and biases
        d_loss__d_w_h = np.dot(delta_h.T, X)
        d_loss__d_b_h = np.sum(delta_h, axis=0)

        return d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h
