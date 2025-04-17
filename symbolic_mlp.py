# symbolic_mlp.py
"""Compute symbolic expressions for a general MLP and its gradients using Sympy."""

import sympy as sp
from IPython.display import display, Math

# Activation functions
def soft_relu(z):
    return sp.log(1 + sp.exp(z))

def tanh(z):
    return (sp.exp(z) - sp.exp(-z)) / (sp.exp(z) + sp.exp(-z))

def sigmoid(z):
    return 1 / (1 + sp.exp(-z))

def mlp_structure(input_size, hidden_layers):
    """
    Define the structure of a MLP using symbolic computation.
    """
    x = sp.symbols(f'x1:{input_size+1}')
    weights = {}
    biases = {}
    prev_layer_size = input_size

    for i, layer_size in enumerate(hidden_layers + [1], start=1):
        for k in range(1, layer_size + 1):
            for j in range(1, prev_layer_size + 1):
                weights[(i, j, k)] = sp.Symbol(f'w_{i}^{j},{k}')
            biases[(i, k)] = sp.Symbol(f'b_{i}^{k}')
        prev_layer_size = layer_size

    return x, weights, biases

def mlp_forward(x, weights, biases, hidden_layers, activation_fn=soft_relu):
    layer_input = list(x)
    for i, layer_size in enumerate(hidden_layers + [1], start=1):
        layer_output = []
        for k in range(1, layer_size + 1):
            neuron_input = sum(
                weights[(i, j, k)] * layer_input[j-1]
                for j in range(1, len(layer_input)+1)
            ) + biases[(i, k)]
            neuron_output = activation_fn(neuron_input)                 if i < len(hidden_layers) + 1 else neuron_input
            layer_output.append(neuron_output)
        layer_input = layer_output
    return layer_input[0]

def calculate_gradients(output, x):
    first_order = [sp.diff(output, xi) for xi in x]
    second_order = [[sp.diff(fg, xj) for xj in x] for fg in first_order]
    return first_order, second_order

if __name__ == "__main__":
    input_size   = 2
    hidden_layers = [2, 3]
    x, weights, biases = mlp_structure(input_size, hidden_layers)
    output = mlp_forward(x, weights, biases, hidden_layers, activation_fn=soft_relu)
    first_order_grads, second_order_grads = calculate_gradients(output, x)

    display(Math("f(\mathbf{x}) = " + sp.latex(output)))
    print("First-order gradients:")
    for i, g in enumerate(first_order_grads, start=1):
        display(Math(f"\frac{{\partial f}}{{\partial x_{i}}} = " + sp.latex(g)))
    print("Second-order gradients:")
    for i, row in enumerate(second_order_grads, start=1):
        for j, g2 in enumerate(row, start=1):
            display(Math(f"\frac{{\partial^2 f}}{{\partial x_{i} \partial x_{j}}} = " + sp.latex(g2)))
