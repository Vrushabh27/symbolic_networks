# symbolic_gnn.py
"""
Compute symbolic expressions for a multi-layer GNN and its gradients using Sympy.
"""

import sympy as sp
from IPython.display import display, Math

def relu(z):
    return sp.Max(0, z)

def gnn_structure(num_nodes, feat_dim, num_layers):
    """
    Create symbolic node features, adjacency, and parameters for a multi-layer GNN.
    Returns:
      X: initial node features
      A: adjacency dict
      W_msg, b_msg, W_up, b_up: dicts mapping layer->parameters
    """
    X = [[sp.Symbol(f'x_{i}_{d}') for d in range(feat_dim)] for i in range(num_nodes)]
    A = {i: [j for j in range(num_nodes) if j != i] for i in range(num_nodes)}

    W_msg = {}
    b_msg = {}
    W_up = {}
    b_up = {}
    for L in range(1, num_layers+1):
        W_msg[L] = [[sp.Symbol(f"Wm_{L}_{i}_{j}") for j in range(feat_dim)] for i in range(2*feat_dim)]
        b_msg[L] = [sp.Symbol(f"bm_{L}_{j}") for j in range(feat_dim)]
        W_up[L] = [[sp.Symbol(f"Wu_{L}_{i}_{j}") for j in range(feat_dim)] for i in range(2*feat_dim)]
        b_up[L] = [sp.Symbol(f"bu_{L}_{j}") for j in range(feat_dim)]
    return X, A, W_msg, b_msg, W_up, b_up

def gnn_forward(X, A, W_msg, b_msg, W_up, b_up, num_layers):
    """
    Forward pass for num_layers of message-passing GNN.
    """
    X_current = X
    for L in range(1, num_layers+1):
        X_next = []
        feat_dim = len(X_current[0])
        def mlp(vec, W, b):
            return [sum(W[k][j]*vec[k] for k in range(len(vec))) + b[j] for j in range(feat_dim)]
        for i in range(len(X_current)):
            agg = [0]*feat_dim
            for j in A[i]:
                inp = X_current[i] + X_current[j]
                m = mlp(inp, W_msg[L], b_msg[L])
                m_act = [relu(v) for v in m]
                agg = [agg[d] + m_act[d] for d in range(feat_dim)]
            upd_inp = X_current[i] + agg
            u = mlp(upd_inp, W_up[L], b_up[L])
            u_act = [relu(v) for v in u]
            X_next.append(u_act)
        X_current = X_next
    return X_current

def calculate_gradients(output, inputs):
    first = [sp.diff(output, x) for x in inputs]
    second = [[sp.diff(f, xj) for xj in inputs] for f in first]
    return first, second

if __name__ == "__main__":
    num_nodes, feat_dim, num_layers = 4, 3, 2
    X, A, Wm, bm, Wu, bu = gnn_structure(num_nodes, feat_dim, num_layers)
    Xp = gnn_forward(X, A, Wm, bm, Wu, bu, num_layers)
    out = Xp[0][0]
    display(Math("x'_{0,0} = " + sp.latex(out)))
    flat_inputs = [X[i][d] for i in range(num_nodes) for d in range(feat_dim)]
    f1, f2 = calculate_gradients(out, flat_inputs)
    print("First-order grads:")
    for idx, g in enumerate(f1):
        display(Math(f"g_{idx} = {sp.latex(g)}"))
    print("Second-order grads:")
    for i,row in enumerate(f2):
        for j,g2 in enumerate(row):
            display(Math(f"h_{{{i},{j}}} = {sp.latex(g2)}"))
