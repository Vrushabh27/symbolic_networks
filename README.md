# Symbolic Neural Networks

This repository contains Python modules that compute symbolic forward passes and gradients for various neural architectures using Sympy:

- `symbolic_mlp.py`: Multilayer Perceptron (MLP)
- `symbolic_lstm.py`: Stacked LSTM network
- `symbolic_transformer.py`: Full Transformer (encoder-decoder) block
- `symbolic_gnn.py`: Configurable multi-layer Graph Neural Network (GNN)

## Requirements

- Python 3.7+
- Sympy
- IPython (for display in notebooks)

## Usage

Run each module in a Jupyter notebook or as a script:

```bash
python symbolic_gnn.py
```

This will display the symbolic expression for the chosen network output and its first- and second-order derivatives with respect to the inputs.
