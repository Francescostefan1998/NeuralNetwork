from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import time
import math

# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.values                  # Convert from pandas to NumPy
y = y.astype(int).values      # Convert labels to integer

# Normalize the pixel values to [-1, 1]
X = ((X / 255.) - 0.5) * 2

# Split the dataset: 10k test, 5k validation, rest training
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=10000, random_state=123, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)

# --- MLP implementation with visualization starts here ---

# Sigmoid activation function
def sigmoid(z):
    return 1. / (1. + np.exp(-z))

# Convert class labels to one-hot encoded vectors
def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary

# Create a visualization for the network
class NetworkVisualizer:
    def __init__(self, input_size, hidden_size, output_size, max_display_nodes=50):
        # Create a larger figure
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.ax.set_title('', fontsize=16)
        self.ax.axis('off')
        
        # For large input layers, we'll only display a subset of nodes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Limit number of displayed input nodes for visualization clarity
        self.display_input_nodes = min(max_display_nodes, input_size)
        
        # Store node positions
        self.input_nodes = []
        self.hidden_nodes = []
        self.output_nodes = []
        
        # Store connections
        self.input_to_hidden_connections = []
        self.hidden_to_output_connections = []
        
        # Initialize visualization
        self._setup_network()
        
        # Custom colormap for activations
        colors = [(0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0)]  # Blue -> White -> Red
        self.activation_cmap = LinearSegmentedColormap.from_list('activation_cmap', colors)
        
        plt.ion()  # Turn on interactive mode
        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.001)
    
    def _setup_network(self):
        # Clear previous plot
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_title('', fontsize=16)
        
        # Layers positions (x-coordinates)
        input_layer_x = 0.15
        hidden_layer_x = 0.5
        output_layer_x = 0.85
        
        # Create nodes for the input layer in a curved arrangement
        self.input_nodes = []
        
    
        # Arrange input nodes in a vertical line filling the plot
        for i in range(self.display_input_nodes):
            y_pos = 1.0 - i / (self.display_input_nodes - 1)  # from 1 to 0
            node = self.ax.add_patch(plt.Circle((input_layer_x, y_pos), 0.008, color='lightgray', zorder=2))
            self.input_nodes.append(node)

        
        # Label for input layer
        self.ax.text(input_layer_x, 0.05, f'Input Layer\n({self.input_size} nodes)', ha='center', fontsize=12)
            
        # Create nodes for the hidden layer
        step_size = 0.8 / (self.hidden_size + 1)
        for i in range(self.hidden_size):
            y_pos = 0.1 + (i + 1) * step_size
            node = self.ax.add_patch(plt.Circle((hidden_layer_x, y_pos), 0.015, color='lightgray', zorder=2))
            self.hidden_nodes.append(node)
            
        # Label for hidden layer
        self.ax.text(hidden_layer_x, 0.05, f'Hidden Layer\n({self.hidden_size} nodes)', ha='center', fontsize=12)
        
        # Create nodes for the output layer
        step_size = 0.8 / (self.output_size + 1)
        for i in range(self.output_size):
            y_pos = 0.1 + (i + 1) * step_size
            node = self.ax.add_patch(plt.Circle((output_layer_x, y_pos), 0.02, color='lightgray', zorder=2))
            self.output_nodes.append(node)
        
        # Label for output layer
        self.ax.text(output_layer_x, 0.05, f'Output Layer\n({self.output_size} nodes)', ha='center', fontsize=12)
        
        # Create connections between layers
        # Connect input to hidden (sparse connections for clarity)
        for i in range(len(self.input_nodes)):
            for j in range(len(self.hidden_nodes)):
                if (i * j) % 5 == 0:  # Display only a subset of connections
                    line = Line2D([self.input_nodes[i].center[0], self.hidden_nodes[j].center[0]],
                                  [self.input_nodes[i].center[1], self.hidden_nodes[j].center[1]],
                                  color='lightgray', linewidth=0.5, alpha=0.3, zorder=1)
                    self.ax.add_line(line)
                    self.input_to_hidden_connections.append(line)
        
        # Connect hidden to output
        for i in range(len(self.hidden_nodes)):
            for j in range(len(self.output_nodes)):
                line = Line2D([self.hidden_nodes[i].center[0], self.output_nodes[j].center[0]],
                              [self.hidden_nodes[i].center[1], self.output_nodes[j].center[1]],
                              color='lightgray', linewidth=0.5, alpha=0.3, zorder=1)
                self.ax.add_line(line)
                self.hidden_to_output_connections.append(line)
                
        # Set plot limits
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
    
    def update_visualization(self, input_activations, hidden_activations, output_activations, weights_ih=None, weights_ho=None):
        # Update input nodes
        display_indices = np.linspace(0, len(input_activations) - 1, len(self.input_nodes), dtype=int)
        for i, idx in enumerate(display_indices):
            activation = input_activations[idx]
            color = self.activation_cmap((activation + 1) / 2)  # Map [-1, 1] to [0, 1]
            self.input_nodes[i].set_color(color)
        
        # Update hidden nodes
        for i, activation in enumerate(hidden_activations):
            color = self.activation_cmap(activation)  # Sigmoid outputs are already [0, 1]
            self.hidden_nodes[i].set_color(color)
        
        # Update output nodes
        for i, activation in enumerate(output_activations):
            color = self.activation_cmap(activation)
            self.output_nodes[i].set_color(color)
        
        # Update connections from input to hidden
        connection_idx = 0
        for i in range(len(self.input_nodes)):
            for j in range(len(self.hidden_nodes)):
                if (i * j) % 5 == 0:  # Match the subset we created
                    if weights_ih is not None:
                        # Color based on signal strength (input activation * weight)
                        input_idx = display_indices[i]
                        weight = weights_ih[j, input_idx]
                        signal = input_activations[input_idx] * weight
                        alpha = min(0.8, abs(signal) * 3)  # Scale for visibility
                        if signal > 0:
                            color = 'red'
                        else:
                            color = 'blue'
                        self.input_to_hidden_connections[connection_idx].set_color(color)
                        self.input_to_hidden_connections[connection_idx].set_alpha(alpha)
                        self.input_to_hidden_connections[connection_idx].set_linewidth(0.5 + abs(signal) * 2)
                    connection_idx += 1
        
        # Update connections from hidden to output
        connection_idx = 0
        for i in range(len(self.hidden_nodes)):
            for j in range(len(self.output_nodes)):
                if weights_ho is not None:
                    # Color based on signal strength (hidden activation * weight)
                    weight = weights_ho[j, i]
                    signal = hidden_activations[i] * weight
                    alpha = min(0.8, abs(signal) * 3)  # Scale for visibility
                    if signal > 0:
                        color = 'red'
                    else:
                        color = 'blue'
                    self.hidden_to_output_connections[connection_idx].set_color(color)
                    self.hidden_to_output_connections[connection_idx].set_alpha(alpha)
                    self.hidden_to_output_connections[connection_idx].set_linewidth(0.5 + abs(signal) * 2)
                connection_idx += 1
        
        # Update the plot
        self.fig.canvas.draw_idle()
        plt.pause(0.002)  # Double the pause time to slow down the animation
        

# Multilayer Perceptron class with visualization
class NeuralNetMLP:
    def __init__(self, num_features, num_hidden, num_classes, random_seed=123, visualize=True):
        super().__init__()
        self.num_classes = num_classes
        self.visualize = visualize
        rng = np.random.RandomState(random_seed)

        # Init weights and biases for hidden layer
        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        # Init weights and biases for output layer
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)
        
        # Initialize visualizer
        if self.visualize:
            self.viz = NetworkVisualizer(num_features, num_hidden, num_classes)

    def forward(self, X, visualize_sample=None):
        # If a specific sample is provided for visualization, use it
        if visualize_sample is None and X.shape[0] > 0:
            visualize_sample = 0  # Visualize the first sample in the batch by default
        
        # Compute activations for hidden layer
        z_h = np.dot(X, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # Compute activations for output layer
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        
        # Update visualization if enabled
        if self.visualize and X.shape[0] > 0:
            sample_input = X[visualize_sample]
            sample_hidden = a_h[visualize_sample]
            sample_output = a_out[visualize_sample]
            self.viz.update_visualization(
                sample_input, 
                sample_hidden, 
                sample_output,
                self.weight_h,
                self.weight_out
            )

        return a_h, a_out

    def backward(self, X, a_h, a_out, y, visualize_sample=None):
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


def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]


def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehot_targets - probas)**2)


def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)


def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas)**2)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss
    mse = mse / (i + 1)
    acc = correct_pred / num_examples
    return mse, acc


def train(model, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate=0.1):
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for e in range(num_epochs):
        # iterate over minibatches
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
        for batch_idx, (X_train_mini, y_train_mini) in enumerate(minibatch_gen):
            # For visualization, select a random sample from the batch
            viz_sample = np.random.randint(0, X_train_mini.shape[0])
            
            # Compute outputs
            a_h, a_out = model.forward(X_train_mini, visualize_sample=viz_sample)
            
            # Slow down the animation with a pause
            time.sleep(0.02)  # Add delay for better visualization
            
            # Compute gradients
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = model.backward(
                X_train_mini, a_h, a_out, y_train_mini, visualize_sample=viz_sample
            )
            
            # Update the weights
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out
            
            # Additional delay for weight updates
            if batch_idx % 10 == 0:  # Update display every 10 batches
                time.sleep(0.02)  # Add more delay for better visibility

        # Epoch logging
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
        train_acc, valid_acc = train_acc * 100, valid_acc * 100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        print(f'Epoch: {e+1:03d}/{num_epochs:03d} | Train MSE: {train_mse:.2f} | Train Acc: {train_acc:.2f}% | Valid Acc: {valid_acc:.2f}%')
    
    return epoch_loss, epoch_train_acc, epoch_valid_acc


# Parameters
num_features = 28*28
num_hidden = 50
num_classes = 10
minibatch_size = 100
num_epochs = 5  # Further reduced for demonstration due to slower animation

# Create and train the model
np.random.seed(123)
model = NeuralNetMLP(num_features=num_features, num_hidden=num_hidden, num_classes=num_classes)

# Check initial performance
_, probas = model.forward(X_valid)
mse = mse_loss(y_valid, probas)
print(f'Initial validation MSE: {mse:.1f}')
predicted_labels = np.argmax(probas, axis=1)
acc = accuracy(y_valid, predicted_labels)
print(f'Initial validation accuracy: {acc*100:.1f}%')

# Train the model
epoch_loss, epoch_train_acc, epoch_valid_acc = train(
    model, X_train, y_train, X_valid, y_valid, 
    num_epochs=num_epochs, learning_rate=0.1
)

# Plot training progress
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), epoch_loss)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), epoch_train_acc, label='Train')
plt.plot(range(1, num_epochs+1), epoch_valid_acc, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# For testing with individual images
def visualize_prediction(model, X_test, y_test, index):
    # Display the image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(X_test[index].reshape(28, 28), cmap='Greys')
    plt.title(f'Test Image (True: {y_test[index]})')
    plt.axis('off')
    
    # Forward pass and display network
    a_h, a_out = model.forward(X_test[index:index+1], visualize_sample=0)
    predicted = np.argmax(a_out[0])
    
    # Display probabilities
    plt.subplot(1, 2, 2)
    plt.bar(range(10), a_out[0])
    plt.xticks(range(10))
    plt.title(f'Prediction: {predicted}')
    plt.tight_layout()
    plt.show()

# Test with a few examples
for i in range(5):
    idx = np.random.randint(0, len(X_test))
    visualize_prediction(model, X_test, y_test, idx)
    plt.pause(2)  # Longer pause to allow viewing

print("Training and visualization complete!")