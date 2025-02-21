# smolGrad
<img src="smolgrad_logo.png" alt="smolGrad Logo" width="100">


smolGrad is a lightweight, from-scratch autograd engine and neural network library written entirely in Python. It implements a simple automatic differentiation system through a custom `Value` class and demonstrates how to build neural network components—such as neurons, layers, and multi-layer perceptrons (MLPs)—without relying on heavyweight frameworks. smolGrad is perfect for learning the fundamentals of backpropagation, understanding computation graphs, and experimenting with neural network design in a clear and minimalistic environment.

## Features

- **Automatic Differentiation:** Custom `Value` class to build computation graphs and perform backpropagation.
- **Neural Network Components:** Easily create neurons, layers, and MLPs from scratch.
- **Educational:** A minimal and clear implementation ideal for understanding the inner workings of autograd.

## Example Usage

### 1. Using the Autograd Engine

The `Value` class builds a computation graph and computes gradients via backpropagation. For example:

```python
from engine import Value

# Create some Value objects with labels (optional)
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

# Perform operations: compute d = a * b + c
d = a * b + c
print("d =", d)  # Expected output: Value(data=calculated_value, grad=0.0)

# Compute gradients using backpropagation
d.backward()

print("Gradient of a:", a.grad)
print("Gradient of b:", b.grad)
```

### 2. Building and Using a Neural Network

The neural network components in this project let you build a multi-layer perceptron (MLP) from scratch. Below is an example of how to create an MLP, pass an input through it, compute a loss, and perform backpropagation to obtain gradients for all parameters.

```python
from nn import MLP
from engine import Value

# Create an MLP with:
# - 3 input neurons
# - One hidden layer with 4 neurons
# - 1 output neuron
model = MLP(3, [4, 1])

# Prepare an input vector (each element is a Value object)
x = [Value(2.0), Value(3.0), Value(-1.0)]

# Forward pass: compute the network output
output = model(x)
print("Network Output:", output)

# Define a target value and compute a simple squared error loss
target = Value(1.0)
loss = (output - target) ** 2
print("Loss:", loss)

# Backward pass: compute gradients for all model parameters
loss.backward()

# Print gradients for each parameter in the model
for param in model.parameters():
    print(param, "gradient:", param.grad)
```
