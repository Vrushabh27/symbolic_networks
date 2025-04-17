# symbolic_lstm.py
"""
Compute symbolic expressions for a stacked LSTM network and its gradients using Sympy.
"""

import sympy as sp
from IPython.display import display, Math

def sigmoid(z):
    return 1/(1 + sp.exp(-z))

def tanh(z):
    return (sp.exp(z) - sp.exp(-z))/(sp.exp(z) + sp.exp(-z))

def lstm_structure(input_size, hidden_layers):
    x = list(sp.symbols(f'x1:{input_size+1}'))
    h_prev = {}
    c_prev = {}
    for L, hsize in enumerate(hidden_layers, start=1):
        h_prev[L] = list(sp.symbols(f'h{L}_prev1:{hsize+1}'))
        c_prev[L] = list(sp.symbols(f'c{L}_prev1:{hsize+1}'))
    gates = ['i','f','c','o']
    W = {}
    U = {}
    b = {}
    for L, hsize in enumerate(hidden_layers, start=1):
        in_size = input_size if L==1 else hidden_layers[L-2]
        for g in gates:
            for j in range(hsize):
                b[(L,g,j)] = sp.Symbol(f'b_{L}{g}{j}')
                for i in range(in_size):
                    W[(L,g,i,j)] = sp.Symbol(f'W_{L}{g}{i}{j}')
                for k in range(hsize):
                    U[(L,g,k,j)] = sp.Symbol(f'U_{L}{g}{k}{j}')
    return x, h_prev, c_prev, W, U, b

def lstm_forward(x, h_prev, c_prev, W, U, b, hidden_layers):
    layer_input = x
    h = {}
    c = {}
    gates = ['i','f','c','o']
    for L, hsize in enumerate(hidden_layers, start=1):
        in_size = len(layer_input)
        z = {g:[] for g in gates}
        for g in gates:
            for j in range(hsize):
                lin = sum(W[(L,g,i,j)]*layer_input[i] for i in range(in_size))
                lin += sum(U[(L,g,k,j)]*h_prev[L][k] for k in range(hsize))
                lin += b[(L,g,j)]
                z[g].append(lin)
        i_t = [sigmoid(z['i'][j]) for j in range(hsize)]
        f_t = [sigmoid(z['f'][j]) for j in range(hsize)]
        c_tilde = [tanh(z['c'][j]) for j in range(hsize)]
        o_t = [sigmoid(z['o'][j]) for j in range(hsize)]
        c[L] = [f_t[j]*c_prev[L][j] + i_t[j]*c_tilde[j] for j in range(hsize)]
        h[L] = [o_t[j]*tanh(c[L][j]) for j in range(hsize)]
        layer_input = h[L]
    return h, c

def calculate_gradients(output, x):
    first = [sp.diff(output, xi) for xi in x]
    second = [[sp.diff(f, xj) for xj in x] for f in first]
    return first, second

if __name__ == "__main__":
    input_size = 2
    hidden_layers = [3,2]
    x, h_prev, c_prev, W, U, b = lstm_structure(input_size, hidden_layers)
    h, c = lstm_forward(x, h_prev, c_prev, W, U, b, hidden_layers)
    final_h = h[len(hidden_layers)][0]
    display(Math("h_t^{(L)} = " + sp.latex(final_h)))
    f1, f2 = calculate_gradients(final_h, x)
    print("First-order grads:")
    for i,g in enumerate(f1, start=1):
        display(Math(f"\partial h/\partial x_{i} = {sp.latex(g)}"))
    print("Second-order grads:")
    for i,row in enumerate(f2,start=1):
        for j,g2 in enumerate(row,start=1):
            display(Math(f"\partial^2 h/\partial x_{i}\partial x_{j} = {sp.latex(g2)}"))
