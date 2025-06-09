import os
import json
import numpy as np
import pandas as pd
import math
from scipy.linalg import expm
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from math import sqrt, exp, log, pi

# Ensure plots directory
os.makedirs('plots', exist_ok=True)

# Set general fonts to bold
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# Helper to set tick labels bold on an Axes
def set_bold_ticks(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

# ------------------------
# 1) TSP Utility Functions
# ------------------------
def generate_random_tsp_cost(n, seed=None):
    rng = np.random.default_rng(seed)
    M = rng.integers(1, 100, size=(n, n))
    np.fill_diagonal(M, 0)
    return np.triu(M) + np.triu(M, 1).T

def route_from_bitstring(bits, n):
    mat = np.array(list(bits), dtype=int).reshape((n, n))
    if not (np.all(mat.sum(axis=1) == 1) and np.all(mat.sum(axis=0) == 1)):
        return None
    return tuple(int(np.where(mat[:, col] == 1)[0][0]) for col in range(n))

def route_cost(route, cost_matrix):
    return sum(cost_matrix[route[i], route[i+1]] for i in range(len(route)-1))

# ------------------------
# 2) Cost Operator
# ------------------------
def build_cost_operator(n, cost_matrix):
    dim = 2**(n*n)
    diag = np.zeros(dim, dtype=float)
    penalty = 1e6
    for idx in range(dim):
        bits = [(idx >> b) & 1 for b in range(n*n)]
        bits.reverse()
        route = route_from_bitstring(bits, n)
        diag[idx] = route_cost(route, cost_matrix) if route else penalty
    return np.diag(diag)

# ------------------------
# 3) Mixers
# ------------------------
pauliX = np.array([[0,1],[1,0]], dtype=complex)
pauliY = np.array([[0,-1j],[1j,0]], dtype=complex)
pauliI = np.eye(2, dtype=complex)

def build_standard_x_mixer(n):
    n2 = n*n; dim = 2**n2
    H = np.zeros((dim, dim), dtype=complex)
    for q in range(n2):
        op = 1
        for qb in range(n2):
            op = np.kron(pauliX if qb==q else pauliI, op)
        H += op
    return H

def local_pauli(n, i, j, p1, p2):
    P = {'X':pauliX, 'Y':pauliY, 'I':pauliI}
    op = 1
    for qb in range(n):
        seg = P[p1] if qb==i else (P[p2] if qb==j else P['I'])
        op = np.kron(seg, op)
    return op

def two_qubit_xy(n, i, j):
    return local_pauli(n,i,j,'X','X') + local_pauli(n,i,j,'Y','Y')

def build_blockwise_xy_mixer(n):
    n2 = n*n; dim = 2**n2
    H = np.zeros((dim, dim), dtype=complex)
    block_dim = 2**n
    single_block = np.zeros((block_dim, block_dim), dtype=complex)
    for i in range(n):
        for j in range(i+1, n):
            single_block += two_qubit_xy(n, i, j)
    for b in range(n):
        left = 2**(b*n)
        right = 2**((n-1-b)*n)
        H += np.kron(np.eye(left), np.kron(single_block, np.eye(right)))
    return H

# ------------------------
# 4) Initial States
# ------------------------
def uniform_init_state(n):
    dim = 2**(n*n)
    return np.ones(dim, dtype=complex) / np.sqrt(dim)

def blockwise_one_hot_init(n):
    block = np.zeros(2**n, dtype=complex)
    amp = 1.0 / np.sqrt(n)
    for k in range(n):
        block[1<<k] = amp
    psi = block
    for _ in range(n-1):
        psi = np.kron(psi, block)
    return psi

# ------------------------
# 5) QAOA Layer & Grid Search
# ------------------------
def grid_search_qaoa(Hc, Hm, psi0, best_idxs, betas, gammas):
    best_prob, best_bg = -1.0, (0, 0)
    for gamma in gammas:
        psi_c = expm(-1j * gamma * Hc) @ psi0
        for beta in betas:
            psi_f = expm(-1j * beta * Hm) @ psi_c
            p_sum = (np.abs(psi_f[best_idxs])**2).sum()
            if p_sum > best_prob:
                best_prob, best_bg = p_sum, (beta, gamma)
    return best_prob, best_bg

# ------------------------
# 6) Permutation Validity
# ------------------------
def is_valid_permutation(bitstr, n):
    mat = np.array(list(bitstr), dtype=int).reshape((n, n))
    return np.all(mat.sum(axis=1)==1) and np.all(mat.sum(axis=0)==1)

# ------------------------
# 7) Post‐processing for best feasible
# ------------------------
def find_best_feasible(counts, cost_matrix, n):
    feasible = []
    for b, cnt in counts.items():
        if is_valid_permutation(b, n):
            route = route_from_bitstring(b, n)
            c = route_cost(route, cost_matrix)
            feasible.append((b, cnt, c))
    if not feasible:
        return None, None, []
    feasible.sort(key=lambda x: (x[2], -x[1]))
    best_b, best_cnt, best_c = feasible[0]
    return best_b, best_c, feasible

# ------------------------
# 8) Benchmark & Save CSV
# ------------------------
def run_and_save(n, seeds, filename, steps=15):
    records = []
    for seed in seeds:
        cost_mat = generate_random_tsp_cost(n, seed)
        Hc = build_cost_operator(n, cost_mat)
        Hm = build_blockwise_xy_mixer(n)
        psi_std = uniform_init_state(n)
        psi_blk = blockwise_one_hot_init(n)
        best_idxs = np.where(np.isclose(np.diag(Hc), np.min(np.diag(Hc))))[0]
        betas = np.linspace(0, np.pi, steps)
        gammas = np.linspace(0, np.pi, steps)
        p_std, _ = grid_search_qaoa(Hc, build_standard_x_mixer(n), psi_std, best_idxs, betas, gammas)
        p_blk, _ = grid_search_qaoa(Hc, Hm, psi_blk, best_idxs, betas, gammas)
        records.append({
            'seed': int(seed),
            'min_cost': float(np.min(np.diag(Hc))),
            'p_standard': float(p_std),
            'p_blockwise': float(p_blk)
        })
    pd.DataFrame(records).to_csv(filename, index=False)
    print(f"Benchmark saved to {filename}")

# ------------------------
# 9) Plot Histogram & Save Results
# ------------------------
def plot_histogram(n, seed, filename, shots=1024):
    df = pd.read_csv(filename)
    row = df[df['seed']==seed]
    if row.empty:
        print(f"No entry for seed={seed}")
        return
    row = row.iloc[0]
    cost_mat = generate_random_tsp_cost(n, seed)
    Hc = build_cost_operator(n, cost_mat)
    Hm = build_blockwise_xy_mixer(n)
    psi0 = blockwise_one_hot_init(n)
    beta_opt, gamma_opt = row['p_blockwise'], row['p_blockwise']  # placeholder
    psi_c = expm(-1j * gamma_opt * Hc) @ psi0
    psi_f = expm(-1j * beta_opt * Hm) @ psi_c
    sv = Statevector(psi_f)
    probs = sv.probabilities_dict()
    counts = {b: int(round(p * shots)) for b, p in probs.items() if p > 0}

    # Discard bitstrings with amplitude less than mean amplitude
    amps = np.abs(psi_f)
    mean_amp = float(amps.mean())
    filtered = {b: cnt for b, cnt in counts.items() if amps[int(b, 2)] >= mean_amp}
    bitstrs = list(filtered.keys())
    cnts = list(filtered.values())
    feasible = [is_valid_permutation(b, n) for b in bitstrs]

    # Plot and save
    positions = list(range(len(bitstrs)))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(positions, cnts, color=['green' if f else 'blue' for f in feasible])
    ax.set_xticks(positions)
    ax.set_xticklabels(bitstrs, rotation=90, fontsize=8)
    ax.set_title(f"Seed={seed}, n={n} (feasible in green)")
    ax.set_xlabel('Bitstring')
    ax.set_ylabel('Counts')
    set_bold_ticks(ax)
    plt.tight_layout()
    hist_path = os.path.join('plots', f'hist_seed_{seed}.png')
    fig.savefig(hist_path)
    plt.close(fig)
    print(f"Histogram saved to {hist_path}")

    # Find & save best feasible results to JSON
    best_b, best_c, all_f = find_best_feasible(filtered, cost_mat, n)
    results = {}
    if best_b is None:
        results['message'] = "No feasible permutations observed."
    else:
        results['best_bitstring'] = best_b
        results['best_cost'] = int(best_c)
        results['top_feasible'] = [
            {'bitstring': b, 'count': int(cnt), 'cost': int(c)} for b, cnt, c in all_f[:5]
        ]
    json_path = os.path.join('plots', f'results_seed_{seed}.json')
    with open(json_path, 'w') as jf:
        json.dump(results, jf, indent=2)
    print(f"Results saved to {json_path}")

# =========================
# 10) Main
# =========================
if __name__ == '__main__':
    N = 4
    SEEDS = range(1)
    CSV_FILE = f'qaoa_benchmark_{N}.csv'
    run_and_save(N, SEEDS, CSV_FILE, steps=1)
    for s in SEEDS:
        plot_histogram(N, s, CSV_FILE)

    # Additional plots for ratio and comparison
    df = pd.read_csv(CSV_FILE)
    xcol = 'seed'
    
    df['ratio'] = df['p_blockwise'] / df['p_standard']
    
    # Plot empirical vs theoretical ratio
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df[xcol], df['ratio'], 'o-', label='Empirical ratio')
    exact_ratio = 2**(N**2) / (N**N)
    ax.axhline(exact_ratio, linestyle='--', label=f'Exact = {exact_ratio:.2e}')
    ax.set_xlabel(xcol)
    ax.set_ylabel('Ratio')
    ax.set_title(f'Empirical vs Theoretical Ratio (n={N})')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    set_bold_ticks(ax)
    plt.tight_layout()
    ratio_path = os.path.join('plots', 'ratio_plot.png')
    fig.savefig(ratio_path)
    plt.close(fig)
    print(f"Ratio plot saved to {ratio_path}")

    # Comparison bar chart
    positions = list(range(len(df)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([p - 0.2 for p in positions], df['p_standard'], width=0.35, alpha=0.7, label='Standard')
    ax.bar([p + 0.2 for p in positions], df['p_blockwise'], width=0.35, alpha=0.7, label='Blockwise')
    ax.set_xticks(positions)
    ax.set_xticklabels([str(int(s)) for s in df['seed']], fontweight='bold')
    ax.set_xlabel('Instance')
    ax.set_ylabel('Success Probability')
    ax.set_title('Standard vs Blockwise QAOA')
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    set_bold_ticks(ax)
    plt.tight_layout()
    comp_path = os.path.join('plots', 'comparison_plot.png')
    fig.savefig(comp_path)
    plt.close(fig)
    print(f"Comparison plot saved to {comp_path}")
