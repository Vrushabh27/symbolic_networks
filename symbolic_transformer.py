# symbolic_transformer.py
"""
Compute symbolic expressions for a full Transformer (encoder-decoder) and its gradients using Sympy.
"""

import sympy as sp
from IPython.display import display, Math

def softmax(vec):
    exps = [sp.exp(v) for v in vec]
    S = sum(exps)
    return [e/S for e in exps]

def relu(z):
    return sp.Max(0, z)

def transformer_structure(src_len, tgt_len, d_model, num_heads, ff_dim, num_layers):
    x_src = [[sp.Symbol(f"xsrc_{t}_{d}") for d in range(d_model)] for t in range(src_len)]
    x_tgt = [[sp.Symbol(f"xtgt_{t}_{d}") for d in range(d_model)] for t in range(tgt_len)]
    d_k = d_model // num_heads
    params = {}
    for L in range(1, num_layers+1):
        # Encoder self-attention params
        for g in ('Q','K','V'):
            for h in range(num_heads):
                params[f"Wenc{g}_{L}_{h}"] = [[sp.Symbol(f"Wenc{g}_{L}_{h}_{i}_{j}") for j in range(d_k)] for i in range(d_model)]
        params[f"WOenc_{L}"] = [[sp.Symbol(f"WOenc_{L}_{i}_{j}") for j in range(d_model)] for i in range(d_model)]
        # Encoder FFN params
        params[f"W1enc_{L}"] = [[sp.Symbol(f"W1enc_{L}_{i}_{j}") for j in range(ff_dim)] for i in range(d_model)]
        params[f"b1enc_{L}"] = [sp.Symbol(f"b1enc_{L}_{j}") for j in range(ff_dim)]
        params[f"W2enc_{L}"] = [[sp.Symbol(f"W2enc_{L}_{i}_{j}") for j in range(d_model)] for i in range(ff_dim)]
        params[f"b2enc_{L}"] = [sp.Symbol(f"b2enc_{L}_{j}") for j in range(d_model)]
        params[f"gamma1enc_{L}"] = [sp.Symbol(f"g1enc_{L}_{j}") for j in range(d_model)]
        params[f"beta1enc_{L}"] = [sp.Symbol(f"b1enc_{L}_{j}") for j in range(d_model)]
        params[f"gamma2enc_{L}"] = [sp.Symbol(f"g2enc_{L}_{j}") for j in range(d_model)]
        params[f"beta2enc_{L}"] = [sp.Symbol(f"b2enc_{L}_{j}") for j in range(d_model)]
        # Decoder self-attention params
        for g in ('Q','K','V'):
            for h in range(num_heads):
                params[f"Wdec{g}_self_{L}_{h}"] = [[sp.Symbol(f"Wdec{g}_self_{L}_{h}_{i}_{j}") for j in range(d_k)] for i in range(d_model)]
        params[f"WOdec_self_{L}"] = [[sp.Symbol(f"WOdec_self_{L}_{i}_{j}") for j in range(d_model)] for i in range(d_model)]
        # Decoder cross-attention params
        for g in ('Q','K','V'):
            for h in range(num_heads):
                params[f"Wdec{g}_cross_{L}_{h}"] = [[sp.Symbol(f"Wdec{g}_cross_{L}_{h}_{i}_{j}") for j in range(d_k)] for i in range(d_model)]
        params[f"WOdec_cross_{L}"] = [[sp.Symbol(f"WOdec_cross_{L}_{i}_{j}") for j in range(d_model)] for i in range(d_model)]
        # Decoder FFN params
        params[f"W1dec_{L}"] = [[sp.Symbol(f"W1dec_{L}_{i}_{j}") for j in range(ff_dim)] for i in range(d_model)]
        params[f"b1dec_{L}"] = [sp.Symbol(f"b1dec_{L}_{j}") for j in range(ff_dim)]
        params[f"W2dec_{L}"] = [[sp.Symbol(f"W2dec_{L}_{i}_{j}") for j in range(d_model)] for i in range(ff_dim)]
        params[f"b2dec_{L}"] = [sp.Symbol(f"b2dec_{L}_{j}") for j in range(d_model)]
        params[f"gamma1dec_{L}"] = [sp.Symbol(f"g1dec_{L}_{j}") for j in range(d_model)]
        params[f"beta1dec_{L}"] = [sp.Symbol(f"b1dec_{L}_{j}") for j in range(d_model)]
        params[f"gamma2dec_{L}"] = [sp.Symbol(f"g2dec_{L}_{j}") for j in range(d_model)]
        params[f"beta2dec_{L}"] = [sp.Symbol(f"b2dec_{L}_{j}") for j in range(d_model)]
    return x_src, x_tgt, params

def transformer_forward(x_src, x_tgt, params, src_len, tgt_len, d_model, num_heads, ff_dim, num_layers):
    d_k = d_model // num_heads

    def layer_norm(vec, gamma, beta):
        mu = sum(vec)/len(vec)
        var = sum((v-mu)**2 for v in vec)/len(vec)
        return [(gamma[i]*(vec[i]-mu)/sp.sqrt(var+1e-6) + beta[i]) for i in range(len(vec))]

    # --- Encoder ---
    E = x_src
    for L in range(1, num_layers+1):
        # self-attention
        head_outs = [[0]*d_model for _ in range(src_len)]
        for h in range(num_heads):
            # compute Q, K, V
            Q = [[sum(params[f"WencQ_{L}_{h}"][i][j]*E[t][i] for i in range(d_model)) for j in range(d_k)] for t in range(src_len)]
            K = [[sum(params[f"WencK_{L}_{h}"][i][j]*E[t][i] for i in range(d_model)) for j in range(d_k)] for t in range(src_len)]
            V = [[sum(params[f"WencV_{L}_{h}"][i][j]*E[t][i] for i in range(d_model)) for j in range(d_k)] for t in range(src_len)]
            for t in range(src_len):
                scores = [sum(Q[t][i]*K[s][i] for i in range(d_k))/sp.sqrt(d_k) for s in range(src_len)]
                alpha = softmax(scores)
                ctx = [sum(alpha[s]*V[s][i] for s in range(src_len)) for i in range(d_k)]
                for i in range(d_k):
                    head_outs[t][h*d_k + i] = ctx[i]
        # combine heads
        Att = [[sum(params[f"WOenc_{L}"][i][j]*head_outs[t][i] for i in range(d_model)) for j in range(d_model)] for t in range(src_len)]
        # add & norm
        E = [layer_norm([E[t][j] + Att[t][j] for j in range(d_model)],
                        params[f"gamma1enc_{L}"], params[f"beta1enc_{L}"])
             for t in range(src_len)]
        # FFN
        FF = []
        for t in range(src_len):
            z1 = [sum(params[f"W1enc_{L}"][i][j]*E[t][i] for i in range(d_model)) + params[f"b1enc_{L}"][j] for j in range(ff_dim)]
            a1 = [relu(z) for z in z1]
            z2 = [sum(params[f"W2enc_{L}"][i][j]*a1[i] for i in range(ff_dim)) + params[f"b2enc_{L}"][j] for j in range(d_model)]
            res = [E[t][j] + z2[j] for j in range(d_model)]
            FF.append(layer_norm(res, params[f"gamma2enc_{L}"], params[f"beta2enc_{L}"]))
        E = FF

    # --- Decoder ---
    D = x_tgt
    for L in range(1, num_layers+1):
        # self-attention on decoder inputs
        head_outs_d1 = [[0]*d_model for _ in range(tgt_len)]
        for h in range(num_heads):
            Qd = [[sum(params[f"WdecQ_self_{L}_{h}"][i][j]*D[t][i] for i in range(d_model)) for j in range(d_k)] for t in range(tgt_len)]
            Kd = [[sum(params[f"WdecK_self_{L}_{h}"][i][j]*D[t][i] for i in range(d_model)) for j in range(d_k)] for t in range(tgt_len)]
            Vd = [[sum(params[f"WdecV_self_{L}_{h}"][i][j]*D[t][i] for i in range(d_model)) for j in range(d_k)] for t in range(tgt_len)]
            for t in range(tgt_len):
                scores = [sum(Qd[t][i]*Kd[s][i] for i in range(d_k))/sp.sqrt(d_k) for s in range(tgt_len)]
                alpha = softmax(scores)
                ctx = [sum(alpha[s]*Vd[s][i] for s in range(tgt_len)) for i in range(d_k)]
                for i in range(d_k):
                    head_outs_d1[t][h*d_k + i] = ctx[i]
        Att_d1 = [[sum(params[f"WOdec_self_{L}"][i][j]*head_outs_d1[t][i] for i in range(d_model)) for j in range(d_model)] for t in range(tgt_len)]
        D = [layer_norm([D[t][j] + Att_d1[t][j] for j in range(d_model)], params[f"gamma1dec_{L}"], params[f"beta1dec_{L}"]) for t in range(tgt_len)]

        # cross-attention: queries from D, keys/values from E
        head_outs_d2 = [[0]*d_model for _ in range(tgt_len)]
        for h in range(num_heads):
            Qc = [[sum(params[f"WdecQ_cross_{L}_{h}"][i][j]*D[t][i] for i in range(d_model)) for j in range(d_k)] for t in range(tgt_len)]
            Kc = [[sum(params[f"WdecK_cross_{L}_{h}"][i][j]*E[s][i] for i in range(d_model)) for j in range(d_k)] for s in range(src_len)]
            Vc = [[sum(params[f"WdecV_cross_{L}_{h}"][i][j]*E[s][i] for i in range(d_model)) for j in range(d_k)] for s in range(src_len)]
            for t in range(tgt_len):
                scores = [sum(Qc[t][i]*Kc[s][i] for i in range(d_k))/sp.sqrt(d_k) for s in range(src_len)]
                alpha = softmax(scores)
                ctx = [sum(alpha[s]*Vc[s][i] for s in range(src_len)) for i in range(d_k)]
                for i in range(d_k):
                    head_outs_d2[t][h*d_k + i] = ctx[i]
        Att_d2 = [[sum(params[f"WOdec_cross_{L}"][i][j]*head_outs_d2[t][i] for i in range(d_model)) for j in range(d_model)] for t in range(tgt_len)]
        D = [layer_norm([D[t][j] + Att_d2[t][j] for j in range(d_model)], params[f"gamma2dec_{L}"], params[f"beta2dec_{L}"]) for t in range(tgt_len)]

        # FFN
        FFd = []
        for t in range(tgt_len):
            z1 = [sum(params[f"W1dec_{L}"][i][j]*D[t][i] for i in range(d_model)) + params[f"b1dec_{L}"][j] for j in range(ff_dim)]
            a1 = [relu(z) for z in z1]
            z2 = [sum(params[f"W2dec_{L}"][i][j]*a1[i] for i in range(ff_dim)) + params[f"b2dec_{L}"][j] for j in range(d_model)]
            res = [D[t][j] + z2[j] for j in range(d_model)]
            FFd.append(layer_norm(res, params[f"gamma2dec_{L}"], params[f"beta2dec_{L}"]))
        D = FFd

    return E, D

def calculate_gradients(output_scalar, inputs):
    first = [sp.diff(output_scalar, x) for x in inputs]
    second = [[sp.diff(f, xj) for xj in inputs] for f in first]
    return first, second

if __name__ == "__main__":
    # Example parameters
    src_len, tgt_len, d_model, num_heads, ff_dim, num_layers = 2, 2, 4, 2, 8, 1
    x_src, x_tgt, params = transformer_structure(src_len, tgt_len, d_model, num_heads, ff_dim, num_layers)
    E, D = transformer_forward(x_src, x_tgt, params, src_len, tgt_len, d_model, num_heads, ff_dim, num_layers)
    # pick decoder final token, first dim as output
    out = D[-1][0]
    display(Math("y = " + sp.latex(out)))
    flat_inputs = [x for row in x_src for x in row] + [x for row in x_tgt for x in row]
    f1, f2 = calculate_gradients(out, flat_inputs)
    print("First-order grads:")
    for i,g in enumerate(f1):
        display(Math(f"g_{i} = {sp.latex(g)}"))
    print("Second-order grads:")
    for i,row in enumerate(f2):
        for j,g2 in enumerate(row):
            display(Math(f"h_{{{i},{j}}} = {sp.latex(g2)}"))
