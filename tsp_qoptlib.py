import os
import itertools
import numpy as np
import pandas as pd
from scipy.linalg import expm
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
import tsplib95
import networkx as nx

# ─────────────────────────────────────────────────────────────────────────────
# Global plot settings: bold fonts
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.weight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold'
})

# ─────────────────────────────────────────────────────────────────────────────
# 0) TSPLIB Utility
# ─────────────────────────────────────────────────────────────────────────────
def load_tsp_cost_matrix(tsp_path: str) -> np.ndarray:
    problem = tsplib95.load(tsp_path)
    G = problem.get_graph()
    return nx.to_numpy_array(G), problem.dimension

# ─────────────────────────────────────────────────────────────────────────────
# 1) Route decoding & cost
# ─────────────────────────────────────────────────────────────────────────────
def route_from_bitstring(bits: list[int], n: int):
    mat = np.array(bits).reshape((n, n))
    if not (np.all(mat.sum(axis=1) == 1) and np.all(mat.sum(axis=0) == 1)):
        return None
    return tuple(int(np.where(mat[:, c] == 1)[0][0]) for c in range(n))

def route_cost(route: tuple[int, ...], M: np.ndarray) -> float:
    return sum(M[route[i], route[i+1]] for i in range(len(route) - 1))

# ─────────────────────────────────────────────────────────────────────────────
# 2) Build cost operator
# ─────────────────────────────────────────────────────────────────────────────
def build_cost_operator(n: int, cost_matrix: np.ndarray) -> np.ndarray:
    dim = 2**(n*n)
    diag = np.zeros(dim, dtype=float)
    penalty = 1e6
    for idx in range(dim):
        bits = [(idx >> b) & 1 for b in range(n*n)][::-1]
        r = route_from_bitstring(bits, n)
        diag[idx] = route_cost(r, cost_matrix) if r else penalty
    return np.diag(diag)

# ─────────────────────────────────────────────────────────────────────────────
# 3) Blockwise XY mixer
# ─────────────────────────────────────────────────────────────────────────────
pauliX = np.array([[0,1],[1,0]], dtype=complex)
pauliY = np.array([[0,-1j],[1j,0]], dtype=complex)
pauliI = np.eye(2, dtype=complex)

def two_qubit_xy(n: int, i: int, j: int) -> np.ndarray:
    P = {'X': pauliX, 'Y': pauliY, 'I': pauliI}
    def local(p1, p2):
        op = 1
        for q in range(n):
            seg = P[p1] if q == i else (P[p2] if q == j else P['I'])
            op = np.kron(seg, op)
        return op
    return local('X','X') + local('Y','Y')

def build_blockwise_xy_mixer(n: int) -> np.ndarray:
    block = np.zeros((2**n, 2**n), dtype=complex)
    for i, j in itertools.combinations(range(n), 2):
        block += two_qubit_xy(n, i, j)
    H = np.zeros((2**(n*n), 2**(n*n)), dtype=complex)
    for b in range(n):
        L = 2**(b*n); R = 2**((n-1-b)*n)
        H += np.kron(np.eye(L), np.kron(block, np.eye(R)))
    return H

# ─────────────────────────────────────────────────────────────────────────────
# 4) One-hot product initial
# ─────────────────────────────────────────────────────────────────────────────
def blockwise_one_hot_init(n: int) -> np.ndarray:
    base = np.zeros(2**n, dtype=complex)
    amp = 1/np.sqrt(n)
    for k in range(n):
        base[1 << k] = amp
    psi = base
    for _ in range(n-1):
        psi = np.kron(psi, base)
    return psi

# ─────────────────────────────────────────────────────────────────────────────
# 5) QAOA grid search p=1
# ─────────────────────────────────────────────────────────────────────────────
def grid_search_qaoa(Hc, Hm, psi0, best_idxs, betas, gammas):
    best_p, best_bg = -1.0, (0.0, 0.0)
    for gamma in gammas:
        psi_c = expm(-1j * gamma * Hc) @ psi0
        for beta in betas:
            psi_f = expm(-1j * beta * Hm) @ psi_c
            s = (np.abs(psi_f[best_idxs])**2).sum()
            if s > best_p:
                best_p, best_bg = s, (beta, gamma)
    return best_p, best_bg

# ─────────────────────────────────────────────────────────────────────────────
# 6) Feasibility & postproc
# ─────────────────────────────────────────────────────────────────────────────
def is_valid_permutation(bs: str, n: int) -> bool:
    mat = np.array(list(bs), dtype=int).reshape((n, n))
    return mat.sum(axis=1).all() == 1 and mat.sum(axis=0).all() == 1

def find_best_feasible(counts, cost_matrix, n):
    feas = []
    for b, cnt in counts.items():
        if is_valid_permutation(b, n):
            r = route_from_bitstring([int(ch) for ch in b], n)
            c = route_cost(r, cost_matrix)
            feas.append((b, cnt, c))
    if not feas:
        return None, None, []
    feas.sort(key=lambda x: (x[2], -x[1]))
    return feas[0][0], feas[0][2], feas

# ─────────────────────────────────────────────────────────────────────────────
# 7) Run on TSPLIB & save CSVs
# ─────────────────────────────────────────────────────────────────────────────
def run_on_tsp(tsp_path, bench_csv, best_csv, steps=20, shots=1024):
    C, n = load_tsp_cost_matrix(tsp_path)
    Hc = build_cost_operator(n, C)
    Hm = build_blockwise_xy_mixer(n)
    psi_blk = blockwise_one_hot_init(n)
    idxs = np.where(np.isclose(np.diag(Hc), np.min(np.diag(Hc))))[0]
    betas = np.linspace(0, np.pi, steps)
    gammas = np.linspace(0, np.pi, steps)

    p_blk, (beta_opt, gamma_opt) = grid_search_qaoa(Hc, Hm, psi_blk, idxs, betas, gammas)
    pd.DataFrame([{
        'tsp_file': tsp_path,
        'n': n,
        'min_cost': float(np.min(np.diag(Hc))),
        'p_blockwise': p_blk,
        'beta_blockwise': beta_opt,
        'gamma_blockwise': gamma_opt
    }]).to_csv(bench_csv, index=False)

    psi_c = expm(-1j * gamma_opt * Hc) @ psi_blk
    psi_f = expm(-1j * beta_opt * Hm) @ psi_c
    sv = Statevector(psi_f)
    probs = sv.probabilities_dict()
    counts = {b: int(round(p * shots)) for b, p in probs.items() if p > 0}

    best_b, best_c, _ = find_best_feasible(counts, C, n)
    pd.DataFrame([{
        'tsp_file': tsp_path,
        'best_bitstring': best_b or '',
        'best_cost': best_c if best_c is not None else np.nan
    }]).to_csv(best_csv, index=False)

    print(f"Wrote {bench_csv} and {best_csv}")

# ─────────────────────────────────────────────────────────────────────────────
# 8) Plot (with filtering & saving)
# ─────────────────────────────────────────────────────────────────────────────
def plot_histogram(n, counts):
    mean_count = np.mean(list(counts.values()))
    filtered = {b: c for b, c in counts.items() if c >= mean_count}
    bs = list(filtered.keys())
    cnt = list(filtered.values())
    feas = [is_valid_permutation(b, n) for b in bs]
    colors = ['green' if f else 'blue' for f in feas]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(bs, cnt, color=colors)
    ax.set_xticks(range(len(bs)))
    ax.set_xticklabels(bs, rotation=90, fontsize=8)

    # Bold tick labels explicitly
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.set_title(f"{n}×{n} one-hot grid (feasible in green)")
    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    fname = f"plots/{n}x{n}_histogram.png"
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved histogram to {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# 9) Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    tsp = "TSP/wi3_1.tsp"
    BENCH = "tsp_benchmark.csv"
    BEST = "tsp_best.csv"

    run_on_tsp(tsp, BENCH, BEST, steps=25, shots=1024)

    C, n = load_tsp_cost_matrix(tsp)
    row = pd.read_csv(BENCH).iloc[0]
    beta_opt, gamma_opt = row['beta_blockwise'], row['gamma_blockwise']
    psi_blk = blockwise_one_hot_init(n)
    psi_c = expm(-1j * gamma_opt * build_cost_operator(n, C)) @ psi_blk
    psi_f = expm(-1j * beta_opt * build_blockwise_xy_mixer(n)) @ psi_c
    counts = {b: int(round(p * 1024)) for b, p in Statevector(psi_f).probabilities_dict().items() if p > 0}

    plot_histogram(n, counts)
