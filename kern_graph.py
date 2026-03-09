"""
kern_graph - GTSAM pose graph for inter-submap optimisation.

Provides a unified PoseGraph class with selectable backend:

  - mode="sim3":  7-DoF Similarity3 nodes (scale + rotation + translation)
  - mode="sl4":  15-DoF SL4 nodes (full projective transformation)

Per-submap nodes: each node stores a transform T_s that maps points from
submap-local coordinates to global coordinates.

    p_global = T_s @ p_local

Convention follows InterGroupPoseEstimation:
  - T_0 = Identity (first submap defines the global frame)
  - T_s = T_{prev} @ T_rel  (chain via inter-submap alignment)
  - BetweenFactor measurement = T_rel  (maps curr-local → prev-local)
"""

import gtsam
import numpy as np
from gtsam import NonlinearFactorGraph, Values, noiseModel
from gtsam.symbol_shorthand import X
from gtsam import Similarity3, BetweenFactorSimilarity3, PriorFactorSimilarity3
from gtsam import SL4, BetweenFactorSL4, PriorFactorSL4


# ---------------------------------------------------------------------------
# Helpers: convert between 4x4 matrix <-> GTSAM types
# ---------------------------------------------------------------------------

def mat4_to_sim3(M):
    """Convert a 4x4 similarity matrix [sR, t; 0, 1] to gtsam.Similarity3."""
    sR = M[:3, :3]
    t = M[:3, 3]
    s = np.cbrt(np.linalg.det(sR))          # det(sR) = s^3
    R = sR / max(abs(s), 1e-12)
    return Similarity3(gtsam.Rot3(R), gtsam.Point3(t), float(s))


def sim3_to_mat4(sim):
    """Convert gtsam.Similarity3 to a 4x4 matrix."""
    return sim.matrix()


def invert_sim3_mat4(T):
    """Invert a 4x4 SIM(3) matrix  T = [sR, t; 0, 1].

    T_inv = [R^T/s, -R^T t / s; 0, 1]
    """
    sR = T[:3, :3]
    t = T[:3, 3]
    s = np.cbrt(np.linalg.det(sR))
    R = sR / max(abs(s), 1e-12)
    s_inv = 1.0 / max(abs(s), 1e-12)
    R_inv = R.T
    T_inv = np.eye(4)
    T_inv[:3, :3] = s_inv * R_inv
    T_inv[:3, 3] = -s_inv * (R_inv @ t)
    return T_inv


def normalize_to_sl4(H):
    """Normalize a 4x4 matrix to unit determinant (SL4)."""
    det = np.linalg.det(H)
    if abs(det) < 1e-15:
        return H
    sign = 1.0 if det > 0 else -1.0
    return sign * H / (abs(det) ** 0.25)


# ---------------------------------------------------------------------------
# Pose graph  (one node per submap)
# ---------------------------------------------------------------------------

_DOF = {"sim3": 7, "sl4": 15}


class PoseGraph:
    """Per-submap pose graph with selectable SIM(3) or SL(4) backend.

    mode="sim3":  7-DoF Similarity3 nodes — for well-calibrated submaps.
    mode="sl4":  15-DoF SL4 nodes — for projectively-distorted submaps.

    Both modes expose the same external API.
    """

    def __init__(self, mode: str = "sim3"):
        if mode not in _DOF:
            raise ValueError(f"Unknown graph mode '{mode}'. Choose from {list(_DOF)}.")
        self.mode = mode
        dof = _DOF[mode]

        self.graph = NonlinearFactorGraph()
        self.values = Values()

        sigma = 0.05 * np.ones(dof, dtype=float)
        self.inter_submap_noise = noiseModel.Diagonal.Sigmas(sigma)
        self.anchor_noise = noiseModel.Diagonal.Sigmas(
            np.array([1e-6] * dof, dtype=float)
        )

        self.initialized_nodes = set()
        self.num_loop_closures = 0

    # ---- internal conversions -----------------------------------------------

    def _to_gtsam(self, M):
        """Convert 4x4 matrix to the appropriate GTSAM type."""
        if self.mode == "sim3":
            return mat4_to_sim3(M)
        return SL4(normalize_to_sl4(M))

    def _from_gtsam(self, sym):
        """Read 4x4 matrix from a GTSAM value."""
        if self.mode == "sim3":
            return sim3_to_mat4(self.values.atSimilarity3(sym))
        return self.values.atSL4(sym).matrix()

    # ---- node insertion --------------------------------------------------

    def add_submap_node(self, submap_id, T_local_to_global):
        """Add a submap node with initial local-to-global transform.

        Args:
            submap_id: Integer submap identifier.
            T_local_to_global: 4x4 transform matrix.
        """
        sym = X(submap_id)
        if sym in self.initialized_nodes:
            print(f"[PoseGraph] node {submap_id} already exists.")
            return
        self.values.insert(sym, self._to_gtsam(T_local_to_global))
        self.initialized_nodes.add(sym)

    # ---- factors ---------------------------------------------------------

    def add_between_factor(self, submap_id_1, submap_id_2, T_rel, noise=None):
        """Add a relative constraint between two submap nodes.

        Args:
            submap_id_1: Source submap id.
            submap_id_2: Target submap id.
            T_rel: 4x4 relative transform (maps submap_2 local → submap_1 local).
            noise: Optional noise model (defaults to inter_submap_noise).
        """
        s1 = X(submap_id_1)
        s2 = X(submap_id_2)
        if s1 not in self.initialized_nodes or s2 not in self.initialized_nodes:
            raise ValueError(
                f"Both submaps {submap_id_1} and {submap_id_2} must exist "
                f"before adding a factor.")
        if noise is None:
            noise = self.inter_submap_noise
        measurement = self._to_gtsam(T_rel)
        if self.mode == "sim3":
            self.graph.add(BetweenFactorSimilarity3(s1, s2, measurement, noise))
        else:
            self.graph.add(BetweenFactorSL4(s1, s2, measurement, noise))

    def add_anchor(self, submap_id, T_local_to_global):
        """Add a prior factor to pin a submap's transform."""
        sym = X(submap_id)
        if sym not in self.initialized_nodes:
            raise ValueError(
                f"Trying to add anchor for submap {submap_id} but it is not in the graph.")
        measurement = self._to_gtsam(T_local_to_global)
        if self.mode == "sim3":
            self.graph.add(PriorFactorSimilarity3(sym, measurement, self.anchor_noise))
        else:
            self.graph.add(PriorFactorSL4(sym, measurement, self.anchor_noise))

    # ---- queries ---------------------------------------------------------

    def get_submap_transform(self, submap_id):
        """Get the optimised 4x4 local-to-global transform for a submap."""
        sym = X(submap_id)
        return self._from_gtsam(sym)

    # ---- optimisation ----------------------------------------------------

    def optimize(self, verbose=False):
        """Run Levenberg-Marquardt optimisation on the factor graph."""
        params = gtsam.LevenbergMarquardtParams()
        if verbose:
            params.setVerbosityLM("SUMMARY")
            params.setVerbosity("ERROR")

        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.values, params)

        initial_error = self.graph.error(self.values)
        print(f"Initial total error: {initial_error:.6f}")

        if verbose:
            for i in range(self.graph.size()):
                factor = self.graph.at(i)
                try:
                    e = factor.error(self.values)
                    keys = [gtsam.DefaultKeyFormatter(k)
                            for k in factor.keys()]
                    print(f"  Factor {i:3d}: error = {e:.6f}  keys={keys}")
                except RuntimeError as ex:
                    print(f"  Factor {i:3d}: error could not be computed ({ex})")

        result = optimizer.optimize()

        if verbose:
            for i in range(self.graph.size()):
                factor = self.graph.at(i)
                try:
                    e = factor.error(result)
                    print(f"  Factor {i:3d}: error = {e:.6f}")
                except RuntimeError as ex:
                    print(f"  Factor {i:3d}: error could not be computed ({ex})")

        self.values = result

    # ---- utilities -------------------------------------------------------

    def print_estimates(self):
        for key in sorted(self.initialized_nodes):
            nid = gtsam.symbolIndex(key)
            mat = self.get_submap_transform(nid)
            print(f"Submap {nid}:\n{mat}\n")

    def increment_loop_closure(self):
        self.num_loop_closures += 1

    def get_num_loops(self):
        return self.num_loop_closures

    def get_num_nodes(self):
        return len(self.initialized_nodes)
