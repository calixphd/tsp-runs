"""
pc-qaoa_tsp.py  ––  block-wise permutation-constrained QAOA
p = 1  *and*  p = 2 (grid search)

Author: <you>
"""
# ─────────────────────────────────────────────────────────────────────────────
# Imports & global plot style
# ─────────────────────────────────────────────────────────────────────────────
import os, itertools, numpy as np, pandas as pd
from scipy.linalg import expm
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
import tsplib95, networkx as nx

plt.rcParams.update({
    'font.weight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold'
})

# ─────────────────────────────────────────────────────────────────────────────
# 0)  TSPLIB loader  ↦  dense cost matrix
# ─────────────────────────────────────────────────────────────────────────────
def load_tsp_cost_matrix(tsp_path: str):
    problem   = tsplib95.load(tsp_path)
    graph     = problem.get_graph()
    return nx.to_numpy_array(graph), problem.dimension

# ─────────────────────────────────────────────────────────────────────────────
# 1)  Utilities –– permutation decoding & route cost
# ─────────────────────────────────────────────────────────────────────────────
def route_from_bitstring(bits, n):
    m = np.asarray(bits).reshape(n, n)
    if not (np.all(m.sum(0) == 1) and np.all(m.sum(1) == 1)):
        return None
    return tuple(int(np.where(m[:, c] == 1)[0][0]) for c in range(n))

def route_cost(route, M):
    return sum(M[route[i], route[i + 1]] for i in range(len(route) - 1))

# ─────────────────────────────────────────────────────────────────────────────
# 2)  Problem Hamiltonian  Hc  (diagonal)
# ─────────────────────────────────────────────────────────────────────────────
def build_cost_operator(n, cost_matrix, penalty=1e6):
    dim  = 2 ** (n * n)
    diag = np.empty(dim)
    for idx in range(dim):
        bits   = [(idx >> k) & 1 for k in range(n * n)][::-1]
        route  = route_from_bitstring(bits, n)
        diag[idx] = route_cost(route, cost_matrix) if route else penalty
    return np.diag(diag)

# ─────────────────────────────────────────────────────────────────────────────
# 3)  Block-wise XY-mixer Hamiltonian  Hm
# ─────────────────────────────────────────────────────────────────────────────
PX, PY, I2 = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.eye(2)

def two_qubit_xy(nq, i, j):
    def kron_for(q, op_i, op_j):
        out = 1
        for qbit in range(nq):
            op = op_i if qbit == i else op_j if qbit == j else I2
            out = np.kron(op, out)
        return out
    return kron_for(i, PX, PX) + kron_for(j, PY, PY)

def build_blockwise_xy_mixer(n):
    """\sum_blocks   \sum_{i<j} (X_i X_j + Y_i Y_j)  on each block"""
    block = sum(two_qubit_xy(n, i, j) for i, j in itertools.combinations(range(n), 2))
    H     = np.zeros((2 ** (n * n),) * 2, dtype=complex)
    for b in range(n):
        L = 2 ** (b * n)
        R = 2 ** ((n - 1 - b) * n)
        H += np.kron(np.eye(L), np.kron(block, np.eye(R)))
    return H

# ─────────────────────────────────────────────────────────────────────────────
# 4)  |s₀⟩  =  ⊗_blocks  (1/√n) ∑ₖ |…00100…⟩
# ─────────────────────────────────────────────────────────────────────────────
def blockwise_one_hot_init(n):
    base = np.zeros(2 ** n, dtype=complex)
    base[[1 << k for k in range(n)]] = 1 / np.sqrt(n)
    psi  = base
    for _ in range(n - 1):
        psi = np.kron(psi, base)
    return psi

# ─────────────────────────────────────────────────────────────────────────────
# 5)  Generic “apply-p-layer QAOA” helper
# ─────────────────────────────────────────────────────────────────────────────
def apply_qaoa_state(Hc, Hm, psi0, betas, gammas):
    """Apply layers in order  U(γₖ) U(βₖ) … U(γ₁) U(β₁) |s₀⟩ ."""
    psi = psi0.copy()
    for β, γ in zip(betas, gammas):
        psi = expm(-1j * γ * Hc) @ psi
        psi = expm(-1j * β * Hm) @ psi
    return psi

# ─────────────────────────────────────────────────────────────────────────────
# 6)  Grid-search  p = 1  and  p = 2
# ─────────────────────────────────────────────────────────────────────────────
def grid_search_p1(Hc, Hm, psi0, good_idxs, betas, gammas):
    best, best_par = -1.0, (0.0, 0.0)
    for γ in gammas:
        for β in betas:
            psi = apply_qaoa_state(Hc, Hm, psi0, [β], [γ])
            succ = np.abs(psi[good_idxs]) ** 2
            s = succ.sum()
            if s > best:
                best, best_par = s, (β, γ)
    return best, best_par

def grid_search_p2(Hc, Hm, psi0, good_idxs, betas, gammas):
    best, best_par = -1.0, (0, 0, 0, 0)
    for γ1, γ2 in itertools.product(gammas, repeat=2):
        for β1, β2 in itertools.product(betas, repeat=2):
            psi = apply_qaoa_state(Hc, Hm, psi0, [β1, β2], [γ1, γ2])
            succ = np.abs(psi[good_idxs]) ** 2
            s = succ.sum()
            if s > best:
                best, best_par = s, (β1, γ1, β2, γ2)
    return best, best_par

# ─────────────────────────────────────────────────────────────────────────────
# 7)  Feasibility helpers
# ─────────────────────────────────────────────────────────────────────────────
def is_valid(bs, n):
    m = np.fromiter(map(int, bs), int).reshape(n, n)
    return np.all(m.sum(0) == 1) and np.all(m.sum(1) == 1)

def find_best_feasible(counts, C, n):
    feas = []
    for b, c in counts.items():
        if is_valid(b, n):
            route = route_from_bitstring(list(map(int, b)), n)
            feas.append((b, c, route_cost(route, C)))
    if not feas:
        return None, np.nan
    feas.sort(key=lambda x: (x[2], -x[1]))
    return feas[0][0], feas[0][2]

# ─────────────────────────────────────────────────────────────────────────────
# 8)  Complete pipeline for one TSP file
# ─────────────────────────────────────────────────────────────────────────────
def run_on_tsp(tsp_path, bench_csv, best_csv, steps=21, shots=1024):
    C, n  = load_tsp_cost_matrix(tsp_path)
    Hc    = build_cost_operator(n, C)
    Hm    = build_blockwise_xy_mixer(n)
    psi0  = blockwise_one_hot_init(n)

    best_idxs = np.where(np.isclose(np.diag(Hc), np.min(np.diag(Hc))))[0]
    betas  = np.linspace(0, np.pi, steps)
    gammas = np.linspace(0, np.pi, steps)

    # --- p = 1 -------------------------------------------------------------
    p1_succ, (β1_opt, γ1_opt) = grid_search_p1(Hc, Hm, psi0, best_idxs, betas, gammas)
    psi_p1  = apply_qaoa_state(Hc, Hm, psi0, [β1_opt], [γ1_opt])
    counts1 = {b: int(round(p * shots))
               for b, p in Statevector(psi_p1).probabilities_dict().items() if p > 0}
    best_b1, best_c1 = find_best_feasible(counts1, C, n)

    # --- p = 2 -------------------------------------------------------------
    p2_succ, (β1, γ1, β2, γ2) = grid_search_p2(Hc, Hm, psi0, best_idxs, betas, gammas)
    psi_p2  = apply_qaoa_state(Hc, Hm, psi0, [β1, β2], [γ1, γ2])
    counts2 = {b: int(round(p * shots))
               for b, p in Statevector(psi_p2).probabilities_dict().items() if p > 0}
    best_b2, best_c2 = find_best_feasible(counts2, C, n)

    # --- write CSVs --------------------------------------------------------
    pd.DataFrame([{
        'tsp_file': tsp_path, 'n': n,
        'min_cost': float(np.min(np.diag(Hc))),
        'p1_succ': p1_succ, 'beta1': β1_opt, 'gamma1': γ1_opt,
        'p2_succ': p2_succ, 'beta1_p2': β1, 'gamma1_p2': γ1,
        'beta2_p2': β2, 'gamma2_p2': γ2
    }]).to_csv(bench_csv, index=False)

    pd.DataFrame([{
        'tsp_file': tsp_path,
        'best_p1_bitstring': best_b1 or '',
        'best_p1_cost': best_c1,
        'best_p2_bitstring': best_b2 or '',
        'best_p2_cost': best_c2
    }]).to_csv(best_csv, index=False)

    print(f"✓ Written   {bench_csv}   and   {best_csv}")

    # Optional histogram for p=2
    plot_histogram(n, counts2, tag='p2')

# ─────────────────────────────────────────────────────────────────────────────
# 9)  Histogram (≥ mean count) –– optionally tag with p1 / p2
# ─────────────────────────────────────────────────────────────────────────────
def plot_histogram(n, counts, tag='p1'):
    mean = np.mean(list(counts.values()))
    filt = {b: c for b, c in counts.items() if c >= mean}
    labels, vals = list(filt.keys()), list(filt.values())
    colors = ['green' if is_valid(b, n) else 'blue' for b in labels]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(labels, vals, color=colors)
    ax.set_xticklabels(labels, rotation=90, fontsize=8, weight='bold')
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_weight('bold')
    ax.set_title(f"{n}×{n} one-hot grid, {tag} (feasible = green)", weight='bold')
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    fname = f"plots/{n}x{n}_{tag}_hist.png"
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ↳ histogram saved  →  {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# 10)  Main entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tsp_file = "TSP/wi3_1.tsp"            # <-- path to your TSPLIB file
    run_on_tsp(
        tsp_path   = tsp_file,
        bench_csv  = "tsp_benchmark.csv",
        best_csv   = "tsp_best.csv",
        steps      = 15,                  # grid resolution for β, γ
        shots      = 5024                 # virtual “measurements”
    )
