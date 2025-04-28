from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.values                  # Convert from pandas to NumPy
y = y.astype(int).values      # Convert labels to integer

# Print dataset shapes
# print(X.shape)  # (70000, 784)
# print(y.shape)  # (70000,)

# Normalize the pixel values to [-1, 1]
X = ((X / 255.) - 0.5) * 2
# print(X)

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
import numpy as np

def int_to_onehot(y, num_labels):
    #np.set_printoptions(threshold=np.inf)  # <- This disables summarizing
    #print("---------starts in to onehot------------")
    #print("Labels before one-hot:")
    #print(y)
    
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    
    #print("Labels after one-hot:")
    #print(ary[0])
    #print("-----end in to onehot------")
    return ary


# Multilayer Perceptron class
class NeuralNetMLP:

    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()
        self.num_classes = num_classes
        rng = np.random.RandomState(random_seed)

        # Init weights and biases for hidden layer
        # the following will generate a matrix of weigths basically each line of the matrix will be virtually associated with a virtual neuron and the column of the matrix will be associated to a specific feature
        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        # Init weights and biases for output layer
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, X):
        # Compute activations for hidden layer
        # The dot product that I am getting is actually a matrix, in a single neuron I will get an array, each value would represent the dot product for a specific sample, but here the situation is different. Each array will represent that situation for just one neuron so having multiple neurons that will end up being a matrix
        z_h = np.dot(X, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # Compute activations for output layer
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    # the a_h and the a_out are the value gotten from the sigmoid function, look here above
    # 
    def backward(self, X, a_h, a_out, y):
        # One-hot encode the labels
        y_onehot = int_to_onehot(y, self.num_classes)

        # Compute gradient of loss w.r.t. output
        # The following line calculates the partial derivative of the loss function with respect to the activations of the output layer (a_out)
        # (a_out - y_onehot) This difference represent the error at the output layer for each sample and each class
        # 2. *(...) This multiplication by 2. comes from the derivate of a common loss function used in this type of problem which is the Mean Squared Error(MSE)
        # / y.shape[0] since y.shape[0] gives you the number of samples in the current batch, Dividing by this number of samples is part of the MSE. This ensure that the magnitued of the gradient is not overly dependent on the batch size
        # In summary the d_loss__d_a_out represent the gradient of the Mean Squared Error loss with respect to the output activations, averaged over the number of samples in the batch. I am getting a matrix where each element indicates how much the loss would change with a small change in the corresponding output activation.
        # the following d_loss__d_a_out it is answering to the question Which outputs are contributing most to the error?
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


model = NeuralNetMLP(num_features=28*28, num_hidden=50, num_classes=10)

import numpy as np
num_epochs  = 50
minibatch_size  =100

def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        #print(batch_idx)
        yield X[batch_idx], y[batch_idx] # pauses the function, saves its state, and gives back one item at a time, like a generator.


# iterate over training epochs
for i in range(num_epochs):
    # iterate over minibatches
    minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
    for X_train_mini, y_train_mini in minibatch_gen:
        break
    break

print(X_train_mini.shape)
print(y_train_mini.shape)

def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehot_targets - probas)**2)

def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)

_, probas = model.forward(X_valid)
mse = mse_loss(y_valid, probas)
print(f'Initial validation MSE: {mse:.1f}')
predicted_labels = np.argmax(probas, axis=1)
acc = accuracy(y_valid, predicted_labels)
print(f'Initial validation accuracy: {acc*100:.1f}%')


def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets -probas)**2)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss
    mse = mse/i
    acc = correct_pred/num_examples
    return mse, acc

mse, acc = compute_mse_and_acc(model, X_valid, y_valid)
print(f'Initial valid MSE: {mse:.1f}')
print(f'Initial valid accuracy: {acc*100:.1f}')

def train(model, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate=0.1):
    epoch_loss=[]
    epoch_train_acc=[]
    epoch_valid_acc=[]

    for e in range(num_epochs):
        # iterate over minibatches
        # minibatches will get generated on runtime one by one. Efficient way to empty the memory. Each batch will be generated, will update the weights and then it will disappear and the for loop will proceed with the next batch
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
        for X_train_mini, y_train_mini in minibatch_gen:
            # Compute outputs
            a_h, a_out = model.forward(X_train_mini)
            # Compute gradients
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = model.backward(X_train_mini, a_h, a_out, y_train_mini)
            # Update the weights
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out

        # Epoch logging
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
        train_acc, valid_acc = train_acc * 100, valid_acc * 100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        print(f'Epoch: {e+1:03d}/{num_epochs:03d} | Train MSE: {train_mse:.2f} | Train Acc: {train_acc:.2f}% | Valid Acc: {valid_acc:.2f}%')
    
    return epoch_loss, epoch_train_acc, epoch_valid_acc


np.random.seed(123) 
epoch_loss, epoch_train_acc, epoch_valid_acc = train(model, X_train, y_train, X_valid, y_valid, num_epochs=50, learning_rate=0.1)


# evaluation the network performances
plt.plot(range(len(epoch_loss)), epoch_loss)
plt.ylabel('Mean squared error')
plt.xlabel('Epoch')
plt.show()

plt.plot(range(len(epoch_train_acc)), epoch_train_acc, label='Training')
plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc, label='Validation')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='lower right')
plt.show()

test_mse, test_acc = compute_mse_and_acc(model, X_test, y_test)
print(f'Test accuracy: {test_acc*100:.2f}%')

X_test_subset = X_test[:1000, :]
y_test_subset = y_test[:1000]
_, probas = model.forward(X_test_subset)
test_pred = np.argmax(probas, axis=1)
misclassified_images = X_test_subset[y_test_subset != test_pred][:25]
misclassified_labels = test_pred[y_test_subset != test_pred][:25]
correct_labels = y_test_subset[y_test_subset != test_pred][:25]
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(8, 8))
ax = ax.flatten()
for i in range(25):
    img = misclassified_images[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title(f'{i+1}) '
                    f'True: {correct_labels[i]}\n'
                    f' Predicted: {misclassified_labels[i]}')
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()