"""ndimr's embedding and loadings are PCA's, restricted to the signal components.

    pixi run python tests/pca_equivalence.py

On log-normalized synthetic counts with planted programmes, the embedding must
equal sklearn's PCA scores of the z-scored matrix, and the loadings its
components, up to each component's sign; the loadings must be orthonormal.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("ndimr", ROOT / "ndimr.py")
ndimr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ndimr)


def main():
    rng = np.random.default_rng(1)
    n, g, k = 600, 1500, 5  # cells, genes, planted programmes
    rate = np.exp(rng.normal(0, 1, g)) * (1 + 3 * (rng.random((n, k)) @ (rng.random((k, g)) < 0.05)))
    X = pd.DataFrame(np.log1p(rng.poisson(rate)), index=[f"c{i}" for i in range(n)],
                     columns=[f"g{j}" for j in range(g)])

    rm = ndimr.fit(X)
    emb, load = ndimr.embed(rm)
    k = emb.shape[1]
    assert k > 0, "no signal components"
    assert list(emb.index) == list(X.index) and list(load.index) == list(rm.normal_genes)

    Xg = X[rm.normal_genes]
    Xz = (Xg - Xg.mean()) / Xg.std(ddof=0)
    p = PCA(n_components=k, svd_solver="full").fit(Xz)
    sign = np.sign(np.sum(p.components_.T * load.to_numpy(), axis=0))
    d_emb = np.abs(p.transform(Xz) * sign - emb.to_numpy()).max()
    d_load = np.abs(p.components_.T * sign - load.to_numpy()).max()
    assert d_emb < 1e-8, f"embedding differs from PCA scores by {d_emb:.2e}"
    assert d_load < 1e-8, f"loadings differ from PCA components by {d_load:.2e}"
    assert np.allclose(load.T @ load, np.eye(k)), "loadings not orthonormal"
    print(f"OK: k={k}, max |d| embedding {d_emb:.1e}, loadings {d_load:.1e}")

    # --n_components: n == TW rank is the default path bit for bit; below it, a
    # prefix; above it, still PCA of the z-scored matrix (rm.X was projected
    # onto the signal subspace, so those loadings need the rebuilt Z).
    e, l = ndimr.embed(rm, k, X)
    assert e.equals(emb) and l.equals(load), "n_components == TW rank differs from default"
    e, l = ndimr.embed(rm, k - 2, X)
    assert e.equals(emb.iloc[:, :k - 2]) and l.equals(load.iloc[:, :k - 2])
    m = k + 10
    e, l = ndimr.embed(rm, m, X)
    p = PCA(n_components=m, svd_solver="full").fit(Xz)
    sign = np.sign(np.sum(p.components_.T * l.to_numpy(), axis=0))
    d_emb = np.abs(p.transform(Xz) * sign - e.to_numpy()).max()
    d_load = np.abs(p.components_.T * sign - l.to_numpy()).max()
    assert d_emb < 1e-8 and d_load < 1e-8, f"n={m}: |d| {d_emb:.2e} / {d_load:.2e}"
    assert np.allclose(l.T @ l, np.eye(m)), f"n={m}: loadings not orthonormal"
    print(f"OK: --n_components {k} == default, {k - 2} prefix, {m} == PCA "
          f"(|d| {d_emb:.1e} / {d_load:.1e})")


if __name__ == "__main__":
    main()
