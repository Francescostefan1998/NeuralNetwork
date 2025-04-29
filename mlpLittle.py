import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.animation as animation

# Set random seed for reproducibility
np.random.seed(42)

# Create a meaningful dataset - Flower categorization based on 3 features
# Features: petal length, petal width, sepal ratio
# Output: probability of being flower type A or flower type B

def create_flower_dataset(n_samples=100):
    """
    Create a dataset of flowers with 3 features:
    - Feature 1: Petal length (cm)
    - Feature 2: Petal width (cm)
    - Feature 3: Sepal length/width ratio

    Output:
    - Class 0: Flower type A
    - Class 1: Flower type B
    """
    # Generate data for class 0 (Flower type A)
    n_class0 = n_samples // 2

    # Flower type A tends to have:
    # - Shorter petals (1-3 cm)
    # - Narrower petals (0.5-1.5 cm)
    # - Lower sepal ratio (1.0-2.0)
    class0_X = np.zeros((n_class0, 3))
    class0_X[:, 0] = np.random.uniform(1.0, 3.0, n_class0)  # Petal length
    class0_X[:, 1] = np.random.uniform(0.5, 1.5, n_class0)  # Petal width
    class0_X[:, 2] = np.random.uniform(1.0, 2.0, n_class0)  # Sepal ratio
    class0_y = np.zeros(n_class0)

    # Generate data for class 1 (Flower type B)
    n_class1 = n_samples - n_class0

    # Flower type B tends to have:
    # - Longer petals (2.5-5.0 cm)
    # - Wider petals (1.2-2.5 cm)
    # - Higher sepal ratio (1.8-3.0)
    class1_X = np.zeros((n_class1, 3))
    class1_X[:, 0] = np.random.uniform(2.5, 5.0, n_class1)  # Petal length
    class1_X[:, 1] = np.random.uniform(1.2, 2.5, n_class1)  # Petal width
    class1_X[:, 2] = np.random.uniform(1.8, 3.0, n_class1)  # Sepal ratio
    class1_y = np.ones(n_class1)

    # Combine and shuffle the dataset
    X = np.vstack((class0_X, class1_X))
    y = np.hstack((class0_y, class1_y))

    # Add some noise to make the classification task more interesting
    X += np.random.normal(0, 0.2, X.shape)

    # Shuffle
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    return X, y.astype(int)

# Sigmoid activation function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))  # Clip to avoid overflow

# Derivative of sigmoid
def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Convert integer labels to one-hot encoded vectors
def int_to_onehot(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    for i, val in enumerate(y):
        one_hot[i, val] = 1
    return one_hot

# Neural Network class
class SimpleNeuralNetwork:
    def __init__(self, num_features, num_hidden, num_classes):
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_classes = num_classes

        # Initialize weights and biases with small random values
        self.weight_h = np.random.normal(0, 0.1, (num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        self.weight_out = np.random.normal(0, 0.1, (num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

        # For visualization
        self.current_sample = None
        self.current_z_h = None
        self.current_a_h = None
        self.current_z_out = None
        self.current_a_out = None
        self.current_target = None
        self.current_error = None
        self.weight_updates_h = np.zeros_like(self.weight_h)
        self.weight_updates_out = np.zeros_like(self.weight_out)

    def forward(self, X):
        # Input can be a batch or a single sample
        is_single_sample = X.ndim == 1
        if is_single_sample:
            X = X.reshape(1, -1)

        # Compute hidden layer activations
        z_h = np.dot(X, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # Compute output layer activations
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)

        # Store current activations for visualization
        if is_single_sample:
            self.current_sample = X[0]
            self.current_z_h = z_h[0]
            self.current_a_h = a_h[0]
            self.current_z_out = z_out[0]
            self.current_a_out = a_out[0]

        return a_h, a_out

    def backward(self, X, y, learning_rate=0.01):
        # Ensure X is 2D
        is_single_sample = X.ndim == 1
        if is_single_sample:
            X = X.reshape(1, -1)
            y = np.array([y])

        # Forward pass to get activations (these are already stored by the preceding forward call)
        # a_h, a_out = self.forward(X) # No need to call forward again

        # Convert labels to one-hot encoding
        y_onehot = int_to_onehot(y, self.num_classes)

        # Output layer error
        output_error = self.current_a_out - y_onehot[0] # Use stored activations
        output_delta = output_error * sigmoid_derivative(self.current_z_out)

        # Hidden layer error
        hidden_error = np.dot(output_delta, self.weight_out)
        hidden_delta = hidden_error * sigmoid_derivative(self.current_z_h)

        # Store current error and target for visualization
        if is_single_sample:
            self.current_target = y[0]
            self.current_error = np.mean(output_error ** 2) # Store mean squared error

        # Update weights and biases
        # Reshape activations and deltas for matrix multiplication if needed
        a_h_reshaped = self.current_a_h.reshape(1, -1) if self.current_a_h.ndim == 1 else self.current_a_h
        hidden_delta_reshaped = hidden_delta.reshape(1, -1) if hidden_delta.ndim == 1 else hidden_delta
        output_delta_reshaped = output_delta.reshape(1, -1) if output_delta.ndim == 1 else output_delta
        X_reshaped = X.reshape(1, -1) if X.ndim == 1 else X


        self.weight_updates_out = np.dot(output_delta_reshaped.T, a_h_reshaped)
        self.weight_updates_h = np.dot(hidden_delta_reshaped.T, X_reshaped)

        # --- FIX START ---
        # The shape of weight_updates_out should match the shape of weight_out (num_classes, num_hidden)
        # The shape of weight_updates_h should match the shape of weight_h (num_hidden, num_features)
        # The transpose (.T) was applied incorrectly in the subtraction step.
        self.weight_out -= learning_rate * self.weight_updates_out # Removed .T
        self.bias_out -= learning_rate * np.sum(output_delta_reshaped, axis=0)
        self.weight_h -= learning_rate * self.weight_updates_h # Removed .T
        self.bias_h -= learning_rate * np.sum(hidden_delta_reshaped, axis=0)
        # --- FIX END ---

        # Return the mean squared error
        return self.current_error

# Visualization class
class NeuralNetworkVisualizer:
    def __init__(self, network, X, y, learning_rate=0.05):
        self.network = network
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.current_sample_idx = 0
        self.epoch = 0
        self.errors = []
        self.signal_speed_factor = 0.03 # Controls delay between animation frames
        self.pulse_length_segments = 4 # How many segments make up the pulse

        # Setup the figure and axes
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.suptitle('Neural Network Visualization', fontsize=16)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.axis('off')

        # Initial drawing of the network
        self.neuron_radius = 0.3
        self.layer_spacing = 3

        # Coordinates for neurons
        # Adjusted vertical spacing for better layout
        input_y_coords = np.linspace(2, 8, self.network.num_features)
        hidden_y_coords = np.linspace(2.5, 7.5, self.network.num_hidden)
        output_y_coords = np.linspace(3.5, 6.5, self.network.num_classes)

        self.input_coords = [(2, y) for y in input_y_coords]
        self.hidden_coords = [(5, y) for y in hidden_y_coords]
        self.output_coords = [(8, y) for y in output_y_coords]

        # Initialize neuron circles
        self.input_neurons = [plt.Circle(coord, self.neuron_radius, fill=True, color='lightblue', alpha=0.7) for coord in self.input_coords]
        self.hidden_neurons = [plt.Circle(coord, self.neuron_radius, fill=True, color='lightblue', alpha=0.7) for coord in self.hidden_coords]
        self.output_neurons = [plt.Circle(coord, self.neuron_radius, fill=True, color='lightblue', alpha=0.7) for coord in self.output_coords]

        # Add all neurons to the plot
        for neuron in self.input_neurons + self.hidden_neurons + self.output_neurons:
            self.ax.add_patch(neuron)

        # Initialize base connection lines (always visible, thin)
        self.base_input_hidden_lines = []
        for i, input_coord in enumerate(self.input_coords):
            for j, hidden_coord in enumerate(self.hidden_coords):
                line = Line2D([input_coord[0], hidden_coord[0]],
                              [input_coord[1], hidden_coord[1]],
                              color='gray', alpha=0.3, linewidth=0.5, zorder=0) # Draw below neurons
                self.ax.add_line(line)
                self.base_input_hidden_lines.append((line, i, j))

        self.base_hidden_output_lines = []
        for i, hidden_coord in enumerate(self.hidden_coords):
            for j, output_coord in enumerate(self.output_coords):
                line = Line2D([hidden_coord[0], output_coord[0]],
                              [hidden_coord[1], output_coord[1]],
                              color='gray', alpha=0.3, linewidth=0.5, zorder=0) # Draw below neurons
                self.ax.add_line(line)
                self.base_hidden_output_lines.append((line, i, j))


        # Initialize signal lines for animation (initially empty)
        self.input_hidden_signal_collections = []
        self.input_hidden_segments_data = [] # Store all segments for each connection
        for i, input_coord in enumerate(self.input_coords):
            for j, hidden_coord in enumerate(self.hidden_coords):
                x = np.linspace(input_coord[0], hidden_coord[0], 25) # More points for smoother pulse
                y = np.linspace(input_coord[1], hidden_coord[1], 25)
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                self.input_hidden_segments_data.append(segments)

                signal_collection = LineCollection([], linewidths=[], colors=[], alpha=0.8, zorder=1) # Draw above base lines
                self.ax.add_collection(signal_collection)
                self.input_hidden_signal_collections.append(signal_collection)

        self.hidden_output_signal_collections = []
        self.hidden_output_segments_data = [] # Store all segments for each connection
        for i, hidden_coord in enumerate(self.hidden_coords):
            for j, output_coord in enumerate(self.output_coords):
                x = np.linspace(hidden_coord[0], output_coord[0], 25) # More points for smoother pulse
                y = np.linspace(hidden_coord[1], output_coord[1], 25)
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                self.hidden_output_segments_data.append(segments)

                signal_collection = LineCollection([], linewidths=[], colors=[], alpha=0.8, zorder=1) # Draw above base lines
                self.ax.add_collection(signal_collection)
                self.hidden_output_signal_collections.append(signal_collection)


        # Labels
        self.ax.text(2, 9.2, 'Input Layer', fontsize=12, ha='center')
        self.ax.text(5, 9.2, 'Hidden Layer', fontsize=12, ha='center')
        self.ax.text(8, 9.2, 'Output Layer', fontsize=12, ha='center')

        # Feature names
        feature_names = ['Petal Length', 'Petal Width', 'Sepal Ratio']
        output_names = ['Type A', 'Type B']

        for i, coord in enumerate(self.input_coords):
            self.ax.text(coord[0] - 1.5, coord[1], feature_names[i], fontsize=9, ha='right', va='center')

        for i, coord in enumerate(self.output_coords):
            self.ax.text(coord[0] + 1.5, coord[1], output_names[i], fontsize=9, ha='left', va='center')

        # Neuron value text display
        self.input_values_text = [self.ax.text(coord[0], coord[1], "", fontsize=8, ha='center', va='center', weight='bold')
                                 for coord in self.input_coords]

        self.hidden_values_text = [self.ax.text(coord[0], coord[1], "", fontsize=8, ha='center', va='center', weight='bold')
                                  for coord in self.hidden_coords]

        self.output_values_text = [self.ax.text(coord[0], coord[1], "", fontsize=8, ha='center', va='center', weight='bold')
                                  for coord in self.output_coords]

        # Weight values display
        self.weight_h_texts = []
        text_offset_h = 0.3 # Offset text slightly from the line
        for i, j in [(i, j) for i in range(self.network.num_features) for j in range(self.network.num_hidden)]:
             input_coord = self.input_coords[i]
             hidden_coord = self.hidden_coords[j]
             # Calculate midpoint and an offset perpendicular to the line
             midpoint_x = (input_coord[0] + hidden_coord[0]) / 2
             midpoint_y = (input_coord[1] + hidden_coord[1]) / 2
             dx = hidden_coord[0] - input_coord[0]
             dy = hidden_coord[1] - input_coord[1]
             length = np.sqrt(dx**2 + dy**2)
             if length > 0:
                 # Perpendicular vector (normalized)
                 perp_dx = -dy / length
                 perp_dy = dx / length
                 text_x = midpoint_x + perp_dx * text_offset_h
                 text_y = midpoint_y + perp_dy * text_offset_h
             else: # handle coincident points (shouldn't happen here)
                 text_x, text_y = midpoint_x, midpoint_y

             text = self.ax.text(text_x, text_y, "", fontsize=7, ha='center', va='center',
                                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'), zorder=2) # Draw above lines
             self.weight_h_texts.append((text, i, j))

        self.weight_out_texts = []
        text_offset_out = 0.3 # Offset text slightly from the line
        for i, j in [(i, j) for i in range(self.network.num_hidden) for j in range(self.network.num_classes)]:
             hidden_coord = self.hidden_coords[i]
             output_coord = self.output_coords[j]
             # Calculate midpoint and an offset perpendicular to the line
             midpoint_x = (hidden_coord[0] + output_coord[0]) / 2
             midpoint_y = (hidden_coord[1] + output_coord[1]) / 2
             dx = output_coord[0] - hidden_coord[0]
             dy = output_coord[1] - hidden_coord[1]
             length = np.sqrt(dx**2 + dy**2)
             if length > 0:
                 # Perpendicular vector (normalized)
                 perp_dx = -dy / length
                 perp_dy = dx / length
                 text_x = midpoint_x + perp_dx * text_offset_out
                 text_y = midpoint_y + perp_dy * text_offset_out
             else: # handle coincident points (shouldn't happen here)
                 text_x, text_y = midpoint_x, midpoint_y

             text = self.ax.text(text_x, text_y, "", fontsize=7, ha='center', va='center',
                                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'), zorder=2) # Draw above lines
             self.weight_out_texts.append((text, i, j))


        # Status text
        self.status_text = self.ax.text(5, 0.5, "", fontsize=12, ha='center')
        self.error_text = self.ax.text(5, 9.5, "", fontsize=12, ha='center')

        # Error plot
        self.error_ax = self.fig.add_axes([0.15, 0.05, 0.2, 0.15])
        self.error_ax.set_title('Training Error', fontsize=10)
        self.error_ax.set_xlabel('Iterations', fontsize=8)
        self.error_ax.set_ylabel('MSE', fontsize=8)
        self.error_line, = self.error_ax.plot([], [], 'r-')

        # Initial update of static elements and values
        self.update_static_visualization()
        self.update_neuron_values()
        self.update_weights_text()


    def update_static_visualization(self):
        """Update visualization elements that don't change during animation (neurons, base lines, text positions)"""
         # Update weights between input and hidden layer (base lines alpha)
        for line, i, j in self.base_input_hidden_lines:
            weight = self.network.weight_h[j, i]
            weight_abs = abs(weight)
            line.set_alpha(min(0.5, max(0.1, weight_abs * 2))) # Alpha based on weight magnitude

        # Update weights between hidden and output layer (base lines alpha)
        for line, i, j in self.base_hidden_output_lines:
            weight = self.network.weight_out[j, i]
            weight_abs = abs(weight)
            line.set_alpha(min(0.5, max(0.1, weight_abs * 2))) # Alpha based on weight magnitude


    def update_neuron_values(self):
        """Update the text display and color for neuron values and activations"""
        X_sample = self.X[self.current_sample_idx]
        y_sample = self.y[self.current_sample_idx] # Target is needed for output highlight

        # Update input values text and color
        for i, val in enumerate(X_sample):
             self.input_values_text[i].set_text(f"{val:.2f}")
             # Color input neurons based on value magnitude
             # Ensure X_train is accessible and handle empty X_train if necessary
             if X_train is not None and len(X_train) > 0:
                 max_range = (X_train.max(axis=0) - X_train.min(axis=0)).max() or 1.0 # Handle case where range is zero
                 intensity = min(1.0, max(0.2, abs(val) / max_range * 2.0 )) # Scale by max feature range
             else:
                 intensity = 0.7 # Default intensity

             if val >= 0:
                 self.input_neurons[i].set_facecolor((0.7, 0.7, 1.0, intensity)) # Blueish for positive
             else:
                 self.input_neurons[i].set_facecolor((1.0, 0.7, 0.7, intensity)) # Reddish for negative


        # Update hidden values text (using stored current_a_h)
        if self.network.current_a_h is not None:
             for i, val in enumerate(self.network.current_a_h):
                 self.hidden_values_text[i].set_text(f"{val:.2f}")
                 intensity = min(1.0, max(0.2, val * 1.5)) # Adjust intensity scaling for activations (0-1)
                 self.hidden_neurons[i].set_facecolor((0.7, 1.0, 0.7, intensity)) # Greenish for activation

        # Update output values text (using stored current_a_out)
        if self.network.current_a_out is not None:
             for i, val in enumerate(self.network.current_a_out):
                 self.output_values_text[i].set_text(f"{val:.2f}")
                 intensity = min(1.0, max(0.2, val * 1.5)) # Adjust intensity scaling for activations (0-1)
                 self.output_neurons[i].set_facecolor((1.0, 0.7, 0.7, intensity)) # Reddish for activation

                 # Highlight correct class neuron outline
                 if i == y_sample:
                     self.output_neurons[i].set_edgecolor('green')
                     self.output_neurons[i].set_linewidth(2)
                 else:
                     self.output_neurons[i].set_edgecolor('black')
                     self.output_neurons[i].set_linewidth(1)

    def update_weights_text(self):
         """Update the text display for weight values"""
         for text, i, j in self.weight_h_texts:
             weight = self.network.weight_h[j, i]
             text.set_text(f"{weight:.2f}")
             # Update text color based on weight sign
             if abs(weight) > 0.5:
                 text.set_color('blue' if weight > 0 else 'red')
             else:
                 text.set_color('black')


         for text, i, j in self.weight_out_texts:
             weight = self.network.weight_out[j, i]
             text.set_text(f"{weight:.2f}")
             # Update text color based on weight sign
             if abs(weight) > 0.5:
                 text.set_color('blue' if weight > 0 else 'red')
             else:
                 text.set_color('black')


    def animate_forward_pass(self):
        """Animate the signal propagation through the network during forward pass."""
        X_sample = self.X[self.current_sample_idx]
        y_sample = self.y[self.current_sample_idx] # Target is needed for output neuron highlight later

        # Ensure neuron values are updated BEFORE animation starts
        self.update_neuron_values()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        time.sleep(self.signal_speed_factor * 5) # Pause to show input values

        # --- Animate Input to Hidden Layer ---
        frames = 30 # More frames for smoother pulse movement
        segments_per_connection_ih = self.input_hidden_segments_data[0].shape[0] # Assuming all connections have same segments
        pulse_length = min(self.pulse_length_segments, segments_per_connection_ih)

        # Calculate max possible weighted input for scaling linewidth
        # Need to consider the range of input values and weights
        # Use max from training data for consistent scaling
        max_abs_input = np.max(np.abs(self.X)) if self.X.size > 0 else 1.0 # Use training data for scaling
        max_abs_weight_h = np.max(np.abs(self.network.weight_h)) if self.network.weight_h.size > 0 else 1.0
        max_possible_weighted_input_ih = max_abs_input * max_abs_weight_h * self.network.num_features # Rough upper bound
        if max_possible_weighted_input_ih == 0: max_possible_weighted_input_ih = 1.0


        k_counter_ih = 0
        for frame in range(frames + pulse_length): # Animate until the pulse is off the connection
            k_counter_ih = 0
            for i in range(self.network.num_features):
                 for j in range(self.network.num_hidden):
                    signal_collection = self.input_hidden_signal_collections[k_counter_ih]
                    segments = self.input_hidden_segments_data[k_counter_ih]

                    input_activation = X_sample[i]
                    weight = self.network.weight_h[j, i]
                    weighted_input = input_activation * weight

                    # Only animate if the signal is significant
                    if abs(weighted_input) > 0.01: # Threshold for animation
                        color = 'blue' if weighted_input > 0 else 'red'
                        # Scale linewidth based on the magnitude of the weighted input
                        linewidth = max(1.0, min(6.0, abs(weighted_input) / max_possible_weighted_input_ih * 10.0)) # Scaled by max possible and adjusted multiplier

                        # Determine the segments for the current pulse position
                        start_segment_idx = frame - pulse_length
                        end_segment_idx = frame
                        active_segments = segments[max(0, start_segment_idx) : min(segments_per_connection_ih, end_segment_idx)]

                        if len(active_segments) > 0:
                            signal_collection.set_segments(active_segments)
                            signal_collection.set_color(color)
                            signal_collection.set_linewidth(linewidth)
                            signal_collection.set_alpha(0.8)
                        else:
                            # Clear the signal once it's past
                            signal_collection.set_segments([])
                            signal_collection.set_alpha(0.0) # Hide the collection

                    else:
                         # Clear the signal if not significant
                         signal_collection.set_segments([])
                         signal_collection.set_alpha(0.0) # Hide the collection

                    k_counter_ih += 1

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            time.sleep(self.signal_speed_factor)

        # Ensure all input-hidden signals are cleared after animation
        for signal_collection in self.input_hidden_signal_collections:
             signal_collection.set_segments([])
             signal_collection.set_alpha(0.0) # Hide the collection


        # Update hidden neuron display after input-hidden animation finishes
        self.update_neuron_values() # Update neuron values and highlights
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        time.sleep(self.signal_speed_factor * 5) # Pause to show hidden activation


        # --- Animate Hidden to Output Layer ---
        frames = 30
        segments_per_connection_ho = self.hidden_output_segments_data[0].shape[0]
        pulse_length = min(self.pulse_length_segments, segments_per_connection_ho)

        # Need hidden activations from the completed forward pass
        hidden_activations = self.network.current_a_h

         # Calculate max possible weighted input for scaling linewidth
        max_abs_hidden_activation = np.max(self.network.current_a_h) if self.network.current_a_h is not None and self.network.current_a_h.size > 0 else 1.0 # Max activation is 1 for sigmoid
        max_abs_weight_out = np.max(np.abs(self.network.weight_out)) if self.network.weight_out.size > 0 else 1.0
        max_possible_weighted_input_ho = max_abs_hidden_activation * max_abs_weight_out * self.network.num_hidden # Rough upper bound
        if max_possible_weighted_input_ho == 0: max_possible_weighted_input_ho = 1.0

        k_counter_ho = 0
        for frame in range(frames + pulse_length):
            k_counter_ho = 0
            for i in range(self.network.num_hidden):
                for j in range(self.network.num_classes):
                    signal_collection = self.hidden_output_signal_collections[k_counter_ho]
                    segments = self.hidden_output_segments_data[k_counter_ho]

                    hidden_activation = hidden_activations[i]
                    weight = self.network.weight_out[j, i]
                    weighted_input = hidden_activation * weight

                    # Only animate if the signal is significant
                    if abs(weighted_input) > 0.01: # Threshold for animation
                        color = 'blue' if weighted_input > 0 else 'red'
                         # Scale linewidth based on the magnitude of the weighted input
                        linewidth = max(1.0, min(6.0, abs(weighted_input) / max_possible_weighted_input_ho * 10.0)) # Scaled by max possible and adjusted multiplier

                        # Determine the segments for the current pulse position
                        start_segment_idx = frame - pulse_length
                        end_segment_idx = frame
                        active_segments = segments[max(0, start_segment_idx) : min(segments_per_connection_ho, end_segment_idx)]

                        if len(active_segments) > 0:
                            signal_collection.set_segments(active_segments)
                            signal_collection.set_color(color)
                            signal_collection.set_linewidth(linewidth)
                            signal_collection.set_alpha(0.8)
                        else:
                            # Clear the signal once it's past
                            signal_collection.set_segments([])
                            signal_collection.set_alpha(0.0) # Hide the collection
                    else:
                         # Clear the signal if not significant
                         signal_collection.set_segments([])
                         signal_collection.set_alpha(0.0) # Hide the collection

                    k_counter_ho += 1

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            time.sleep(self.signal_speed_factor)

        # Ensure all hidden-output signals are cleared after animation
        for signal_collection in self.hidden_output_signal_collections:
             signal_collection.set_segments([])
             signal_collection.set_alpha(0.0) # Hide the collection


        # Update output neuron display and highlights after hidden-output animation finishes
        self.update_neuron_values() # This also highlights the target output neuron
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        time.sleep(self.signal_speed_factor * 5) # Pause to show output activation and target


    def train_step(self):
        """Perform one step of training (backward pass, update) without animation."""
        # This method is now just for the non-visual update part of a step
        X_sample = self.X[self.current_sample_idx]
        y_sample = self.y[self.current_sample_idx]

        # Perform backward pass to update weights
        error = self.network.backward(X_sample, y_sample, self.learning_rate)
        self.errors.append(error)

        # Update static visualization elements (weights, error plot)
        self.update_static_visualization()
        # Neuron values and weights text are updated during the animation and after backward pass
        self.update_neuron_values()
        self.update_weights_text()

        # Update error plot
        self.error_line.set_data(range(len(self.errors)), self.errors)
        self.error_ax.relim()
        self.error_ax.autoscale_view()

        # Update status text
        self.status_text.set_text(f"Epoch: {self.epoch+1}, Sample: {self.current_sample_idx+1}/{len(self.X)}")
        if len(self.errors) > 0:
             self.error_text.set_text(f"Current Error: {self.errors[-1]:.4f}")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        return error


    def train(self, epochs=5, sample_delay=0.5):
        """Train the network with visualization and animation for each sample."""
        total_samples = len(self.X) * epochs
        sample_counter = 0

        plt.show(block=False) # Show the initial plot

        while self.epoch < epochs:
            # Get current sample
            X_sample = self.X[self.current_sample_idx]
            y_sample = self.y[self.current_sample_idx]

            # 1. Forward pass (stores intermediate values)
            self.network.forward(X_sample)

            # 2. Animate the forward pass
            self.animate_forward_pass()

            # 3. Perform backward pass and update weights
            error = self.network.backward(X_sample, y_sample, self.learning_rate)
            self.errors.append(error)

            # 4. Update static visualization elements (weights, error plot, neuron values after update)
            # Note: Neuron values will now reflect the result of the *last* forward pass before the weight update
            # The animate_forward_pass already updates neuron values at the end, but this ensures they are updated after backward pass too
            self.update_static_visualization()
            self.update_neuron_values() # Update neuron values and highlights
            self.update_weights_text() # Update displayed weights

            # Update error plot
            self.error_line.set_data(range(len(self.errors)), self.errors)
            self.error_ax.relim()
            self.error_ax.autoscale_view()

            # Update status text
            self.status_text.set_text(f"Epoch: {self.epoch+1}, Sample: {self.current_sample_idx+1}/{len(self.X)}")
            if len(self.errors) > 0:
                 self.error_text.set_text(f"Current Error: {self.errors[-1]:.4f}")

            # 5. Refresh plot after backward pass update
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            # 6. Move to next sample
            self.current_sample_idx = (self.current_sample_idx + 1) % len(self.X)

            # 7. Update epoch counter if we've gone through all samples
            if self.current_sample_idx == 0:
                 self.epoch += 1

            sample_counter += 1

            # Print progress
            if sample_counter % 10 == 0:
                print(f"Progress: {sample_counter}/{total_samples}, Error: {error:.4f}")

            # Delay between samples
            # The animation provides a delay, so a shorter sample_delay might be sufficient
            # time.sleep(sample_delay) # Removed this as animation provides the delay

        print("Training complete!")
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Keep the plot window open


# Generate the dataset
X, y = create_flower_dataset(n_samples=100)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features for better training
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
# Handle cases where std is zero (feature has no variance)
X_std[X_std == 0] = 1e-8
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std


# Create and train the network
network = SimpleNeuralNetwork(num_features=3, num_hidden=3, num_classes=2)
visualizer = NeuralNetworkVisualizer(network, X_train, y_train, learning_rate=0.05)

# Train the network with visualization
# sample_delay now controls the delay between samples (which includes the animation time)
visualizer.train(epochs=3, sample_delay=0.1) # Adjusted sample_delay based on animation length

# Evaluate on test set
correct = 0
for i in range(len(X_test)):
    _, probas = network.forward(X_test[i])
    predicted = np.argmax(probas)
    if predicted == y_test[i]:
        correct += 1

test_accuracy = correct / len(X_test)
print(f"Test accuracy: {test_accuracy:.2f}")

# Plot decision regions (for 2D visualization, use first two features)
plt.figure(figsize=(10, 8))
plt.title('Decision Regions (Petal Length vs Petal Width)')
plt.xlabel('Petal Length (normalized)')
plt.ylabel('Petal Width (normalized)')

# Create a mesh grid
h = 0.05  # Step size - increased for faster plotting
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Use average value for the third feature
# Need to use the normalized mean of the third feature from training data
normalized_z = 0 # Since X_train[:, 2] was normalized to have mean 0 and std 1

# Use the model to predict the mesh grid points
mesh_points = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, normalized_z)]
_, Z = network.forward(mesh_points)
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# Plot the training points
for i, color in zip([0, 1], ['blue', 'red']):
    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1],
                c=color, label=f'Class {i}', alpha=0.5, edgecolors='k')

plt.legend()
plt.show()

# If you want to continue with manual training steps after visualization
# def continue_training(visualizer, steps=20, delay=0.5):
#     for _ in range(steps):
#         visualizer.train_step()
#         time.sleep(delay)
#     print("Additional training complete!")

# continue_training(visualizer, steps=20, delay=0.5)