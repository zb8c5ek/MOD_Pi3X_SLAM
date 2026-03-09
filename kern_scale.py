"""
kern_scale - Alignment solvers for inter-submap registration.

Provides:
  - estimate_sim3_umeyama    - closed-form SIM(3) (Umeyama algorithm)
  - estimate_sim3_ransac     - RANSAC wrapper around Umeyama
  - estimate_rigid_kabsch    - rigid SE(3) alignment (Kabsch algorithm)
                               used to cross-check stereo pair rigidity
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# SIM(3) Umeyama solver
# ---------------------------------------------------------------------------

def estimate_sim3_umeyama(X, Y):
    """Umeyama algorithm: find (s, R, t) minimising  sum ||Y_i - (s R X_i + t)||^2.

    Args:
        X: (N, 3) source points.
        Y: (N, 3) target points.

    Returns:
        s:  scalar scale factor.
        R:  (3, 3) rotation matrix.
        t:  (3,) translation vector.
        T:  (4, 4) similarity matrix  [sR, t; 0, 1].
    """
    assert X.shape == Y.shape and X.shape[1] == 3
    n = X.shape[0]

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    # Cross-covariance  (Umeyama convention: Sigma = Yc^T Xc / n)
    sigma = Yc.T @ Xc / n                       # (3, 3)
    U, S, Vt = np.linalg.svd(sigma)

    # Correct for reflection
    d = np.linalg.det(U) * np.linalg.det(Vt)
    D = np.diag([1.0, 1.0, np.sign(d)])

    R = U @ D @ Vt
    var_x = np.sum(Xc ** 2) / n
    s = np.trace(np.diag(S) @ D) / max(var_x, 1e-12)
    t = mu_y - s * R @ mu_x

    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return s, R, t, T


def estimate_sim3_ransac(X, Y, n_iter=1000, inlier_thresh=0.1, min_samples=4,
                         max_eval_pts=50000, early_stop_ratio=0.85):
    """RANSAC wrapper around Umeyama for robust SIM(3) estimation.

    Args:
        X: (N, 3) source points.
        Y: (N, 3) target points.
        n_iter: RANSAC iterations.
        inlier_thresh: max per-point residual to count as inlier.
        min_samples: correspondences per hypothesis (>= 3).
        max_eval_pts: subsample to this many points for per-iteration
                      residual evaluation. Full set used for final refinement.
        early_stop_ratio: stop early if inlier ratio exceeds this.

    Returns:
        s, R, t, T, inlier_mask
    """
    assert X.shape == Y.shape and X.shape[1] == 3
    n = X.shape[0]
    _t0 = time.perf_counter()

    # Subsample for fast per-iteration evaluation
    if n > max_eval_pts:
        eval_idx = np.random.choice(n, max_eval_pts, replace=False)
        X_eval = X[eval_idx]
        Y_eval = Y[eval_idx]
        n_eval = max_eval_pts
    else:
        X_eval = X
        Y_eval = Y
        n_eval = n
        eval_idx = None

    _t_sub = time.perf_counter()

    best_n_inliers = 0
    best_result = None
    early_stop_n = int(early_stop_ratio * n_eval)
    iters_run = 0

    for iters_run in range(1, n_iter + 1):
        idx = np.random.choice(n, min_samples, replace=False)
        try:
            s, R, t, T = estimate_sim3_umeyama(X[idx], Y[idx])
        except Exception:
            continue

        Y_pred = (s * (R @ X_eval.T)).T + t
        residuals = np.linalg.norm(Y_pred - Y_eval, axis=1)
        ni = int((residuals < inlier_thresh).sum())

        if ni > best_n_inliers:
            best_n_inliers = ni
            best_result = (s, R, t, T)
            if ni >= early_stop_n:
                break

    _t_loop = time.perf_counter()

    # Refine on full set using best model
    if best_result is not None and best_n_inliers >= min_samples:
        s, R, t, T = best_result
        # Identify inliers on the FULL point set
        Y_pred_full = (s * (R @ X.T)).T + t
        full_mask = np.linalg.norm(Y_pred_full - Y, axis=1) < inlier_thresh

        # Re-estimate on full inlier set
        if int(full_mask.sum()) >= min_samples:
            s, R, t, T = estimate_sim3_umeyama(X[full_mask], Y[full_mask])
            Y_pred_full = (s * (R @ X.T)).T + t
            full_mask = np.linalg.norm(Y_pred_full - Y, axis=1) < inlier_thresh
            _t_ref = time.perf_counter()
            print(f"    [RANSAC-SIM3] n={n} eval={n_eval} iters={iters_run}/{n_iter} "
                  f"best_inliers={best_n_inliers}/{n_eval} | "
                  f"subsample={_t_sub-_t0:.3f}s loop={_t_loop-_t_sub:.3f}s "
                  f"refine={_t_ref-_t_loop:.3f}s total={_t_ref-_t0:.3f}s")
            return s, R, t, T, full_mask

    # Fallback: use all points
    s, R, t, T = estimate_sim3_umeyama(X, Y)
    _t_ref = time.perf_counter()
    print(f"    [RANSAC-SIM3] FALLBACK n={n} eval={n_eval} iters={iters_run}/{n_iter} | "
          f"subsample={_t_sub-_t0:.3f}s loop={_t_loop-_t_sub:.3f}s "
          f"refine={_t_ref-_t_loop:.3f}s total={_t_ref-_t0:.3f}s")
    return s, R, t, T, np.ones(n, dtype=bool)


# ---------------------------------------------------------------------------
# Kabsch SE(3) rigid solver  (stereo validation)
# ---------------------------------------------------------------------------

def estimate_rigid_kabsch(X, Y):
    """Kabsch algorithm: find rigid (R, t) minimising  sum ||Y_i - (R X_i + t)||^2.

    Used to validate stereo pairs: since cam0 <-> cam1 is a *known* rigid
    transform, a low RMSD confirms that Pi3X's reconstruction is metrically
    consistent for the stereo pair.

    Args:
        X: (N, 3) source points.
        Y: (N, 3) target points.

    Returns:
        R:    (3, 3) rotation matrix.
        t:    (3,) translation vector.
        T:    (4, 4) rigid transform  [R, t; 0, 1].
        rmsd: root-mean-square deviation after alignment.
    """
    assert X.shape == Y.shape and X.shape[1] == 3
    n = X.shape[0]

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    H = Yc.T @ Xc                                # (3, 3)
    U, _S, Vt = np.linalg.svd(H)

    d = np.linalg.det(U) * np.linalg.det(Vt)
    D = np.diag([1.0, 1.0, np.sign(d)])

    R = U @ D @ Vt
    t = mu_y - R @ mu_x

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    Y_pred = (R @ X.T).T + t
    rmsd = np.sqrt(np.mean(np.sum((Y_pred - Y) ** 2, axis=1)))
    return R, t, T, rmsd
