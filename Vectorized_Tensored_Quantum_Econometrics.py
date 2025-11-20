"""
Vectorized Tensored Quantum Econometric System for Irish Agri Policy Modelling
Author: Jit (concept) – Python scaffold by ChatGPT
Dependencies: numpy, pandas
"""

import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable


# ----------------------------------------------------------------------
# 1. POLICY SPACE – TENSORED / "QUANTUM" LAYER
# ----------------------------------------------------------------------

class PolicySpace:
    """
    Quantum-inspired policy state space.

    - Define discrete policy levers, each with discrete levels.
    - Build tensor-product policy space via Kronecker products.
    - Maintain a complex state vector psi over all configurations.
    - Provide expectation of policy features from |psi|^2.
    """

    def __init__(self, levers: Dict[str, List[float]]):
        """
        Parameters
        ----------
        levers : dict
            Keys = lever names (e.g. "fertiliser_subsidy"),
            Values = list of discrete levels (e.g. [0.0, 0.5, 1.0]).
        """
        self.levers = levers
        self.lever_names = list(levers.keys())
        self.level_lists = [levers[name] for name in self.lever_names]

        # Cartesian product of all lever levels: all policy configurations
        self.config_table = self._build_config_table()

        # Dimension of full policy space
        self.dim = len(self.config_table)

        # Initialise psi as equal superposition (complex amplitudes)
        self.psi = self._init_uniform_state()

        # Precompute a simple "feature matrix" for configs:
        # shape: (n_configs, n_features). Here n_features = number of levers.
        self.feature_matrix = self._build_feature_matrix()

    def _build_config_table(self) -> pd.DataFrame:
        """
        Build a table of all policy configurations.
        Each row = one configuration, columns = lever names.
        """
        configs = list(itertools.product(*self.level_lists))
        df = pd.DataFrame(configs, columns=self.lever_names)
        df["config_id"] = np.arange(len(df))
        return df

    def _init_uniform_state(self) -> np.ndarray:
        """
        Initialise psi as uniform superposition over all policy configs.
        """
        dim = len(self.config_table)
        psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        return psi

    def _build_feature_matrix(self) -> np.ndarray:
        """
        Build feature matrix: rows = configs, cols = lever values.
        """
        return self.config_table[self.lever_names].to_numpy(dtype=float)

    @property
    def probabilities(self) -> np.ndarray:
        """
        Probability distribution over policy configs: |psi|^2.
        """
        return np.real(self.psi * np.conjugate(self.psi))

    def normalise(self):
        """
        Renormalise psi to unit length.
        """
        norm = np.linalg.norm(self.psi)
        if norm == 0:
            raise ValueError("Psi has zero norm; cannot normalise.")
        self.psi /= norm

    def collapse_to_config(self, config_id: int):
        """
        "Measurement" of policy state into a single configuration.
        """
        new_psi = np.zeros_like(self.psi)
        new_psi[config_id] = 1.0 + 0.0j
        self.psi = new_psi

    def expectation_policy_features(self) -> np.ndarray:
        """
        E[policy_features] under |psi|^2.

        Returns
        -------
        np.ndarray
            Shape: (n_features,). In our simple case, features = levers.
        """
        probs = self.probabilities[:, None]  # (n_configs, 1)
        exp = np.sum(probs * self.feature_matrix, axis=0)
        return exp  # (n_features,)

    def apply_unitary(self, U: np.ndarray):
        """
        Apply a (pseudo-)unitary operator to psi:
        psi <- U @ psi

        Parameters
        ----------
        U : np.ndarray
            Shape (dim, dim). Should be approx unitary for full fidelity,
            but we don't enforce it here to keep things flexible.
        """
        if U.shape != (self.dim, self.dim):
            raise ValueError(f"U shape {U.shape} incompatible with dim {self.dim}")
        self.psi = U @ self.psi
        self.normalise()

    def random_unitary(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a random orthonormal matrix via QR-decomposition,
        used as a pseudo-unitary operator on policy space.
        """
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(self.dim, self.dim))
        Q, _ = np.linalg.qr(X)
        return Q.astype(np.complex128)

    def kronecker_policy_operator(self,
                                  local_mats: List[np.ndarray]) -> np.ndarray:
        """
        Build a kron-product operator from local matrices per lever.

        Example:
        - Each lever has K_i states and a local operator A_i (K_i x K_i).
        - Combined operator U = ⊗_i A_i   (Kronecker product).
        - Here we assume levers are sized accordingly.

        Parameters
        ----------
        local_mats : List[np.ndarray]
            List of square matrices, each representing a local operator.

        Returns
        -------
        np.ndarray
            Global operator U of shape (dim, dim).
        """
        # We assume each lever corresponds to len(levels_i) states.
        # We only check shapes roughly.
        if len(local_mats) != len(self.level_lists):
            raise ValueError("One local matrix per lever is required.")

        U = local_mats[0]
        for mat in local_mats[1:]:
            U = np.kron(U, mat)
        # Cast to complex
        return U.astype(np.complex128)

    def config_count(self) -> int:
        return self.dim

    def config_id_to_feature_vector(self, config_id: int) -> np.ndarray:
        """Return lever values for a given configuration id."""
        row = self.config_table.loc[self.config_table["config_id"] == config_id, self.lever_names]
        if row.empty:
            raise ValueError(f"Invalid config_id {config_id}")
        return row.to_numpy(dtype=float).flatten()

    def config_id_to_bits(self, config_id: int, n_bits: int) -> np.ndarray:
        """Map config_id to binary vector (LSB-first)."""
        if config_id >= 2 ** n_bits:
            raise ValueError("n_bits too small for given config_id.")
        bits = np.array([(config_id >> i) & 1 for i in range(n_bits)], dtype=int)
        return bits

    def bits_to_config_id(self, bits: np.ndarray) -> int:
        """Map binary vector (LSB-first) back to integer config_id."""
        bits = np.asarray(bits, dtype=int) % 2
        config_id = 0
        for i, b in enumerate(bits):
            if b:
                config_id |= (1 << i)
        return int(config_id)
    
class QUBOModel:
    """
    Simple QUBO container and brute-force solver for small n.

    E(z) = z^T Q z,   z in {0,1}^n

    - You can export Q to real quantum solvers later.
    - Here we keep a brute-force solver for demonstration / small problems.
    """

    def __init__(self, Q: np.ndarray):
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be a square matrix.")
        self.Q = 0.5 * (Q + Q.T)  # symmetrize

    @property
    def n_vars(self) -> int:
        return self.Q.shape[0]

    def energy(self, z: np.ndarray) -> float:
        """
        Compute E(z) = z^T Q z for binary z.
        """
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        return float((z.T @ self.Q @ z).item())

    def brute_force_solve(self) -> Tuple[np.ndarray, float]:
        """
        Exhaustive search over 2^n states.
        Only feasible for small n (e.g., n <= ~20).
        """
        n = self.n_vars
        best_z = None
        best_E = np.inf

        for state in range(1 << n):
            # binary representation
            bits = np.array([(state >> i) & 1 for i in range(n)], dtype=float)
            E = self.energy(bits)
            if E < best_E:
                best_E = E
                best_z = bits

        return best_z, best_E

class QLDPCCode:
    """
    qLDPC-flavoured scaffold using a classical sparse parity-check matrix H.

    H is an m x n binary matrix.
    Codewords x satisfy H x = 0 (mod 2).

    We implement:
    - syndrome computation
    - simple iterative bit-flip decoder (Gallager-style)
    """

    def __init__(self, H: np.ndarray):
        H = np.asarray(H, dtype=int) % 2
        if H.ndim != 2:
            raise ValueError("H must be 2D.")
        self.H = H
        self.m, self.n = H.shape

        # Precompute row/column neighbors for decoding
        self.check_nodes = [np.where(H[i, :] == 1)[0] for i in range(self.m)]

    @staticmethod
    def random_ldpc(
        n: int,
        m: int,
        row_weight: int = 3,
        col_weight: int = 3,
        seed: Optional[int] = None
    ) -> "QLDPCCode":
        """
        Very rough random LDPC generator (no guarantees on rank/structure).
        For experimentation only.
        """
        rng = np.random.default_rng(seed)
        H = np.zeros((m, n), dtype=int)

        # Add row_weight ones per row
        for i in range(m):
            idx = rng.choice(n, size=row_weight, replace=False)
            H[i, idx] = 1

        # Optionally enforce at least col_weight ones per column
        for j in range(n):
            if H[:, j].sum() < col_weight:
                add_rows = rng.choice(m, size=col_weight - int(H[:, j].sum()), replace=False)
                H[add_rows, j] = 1

        return QLDPCCode(H % 2)

    def syndrome(self, x: np.ndarray) -> np.ndarray:
        """
        s = H x (mod 2)
        """
        x = np.asarray(x, dtype=int) % 2
        if x.shape[0] != self.n:
            raise ValueError("x has wrong length.")
        s = (self.H @ x) % 2
        return s

    def is_valid(self, x: np.ndarray) -> bool:
        return np.all(self.syndrome(x) == 0)

    def bit_flip_decode(
        self,
        y: np.ndarray,
        max_iter: int = 50
    ) -> np.ndarray:
        """
        Simple hard-decision iterative bit-flip decoding.

        y: received bits (0/1).
        returns: decoded bits (0/1).
        """
        x = np.asarray(y, dtype=int) % 2

        for _ in range(max_iter):
            s = self.syndrome(x)
            if np.all(s == 0):
                break  # already a valid codeword

            # For each variable node j, count unsatisfied checks it participates in
            unsatisfied = np.where(s == 1)[0]
            if len(unsatisfied) == 0:
                break

            flip_scores = np.zeros(self.n, dtype=int)
            for check_idx in unsatisfied:
                for var_idx in self.check_nodes[check_idx]:
                    flip_scores[var_idx] += 1

            # Flip those variable nodes that participate in "many" unsatisfied checks
            threshold = max(1, int(np.max(flip_scores) // 2))
            flip_mask = flip_scores >= threshold
            if not np.any(flip_mask):
                break

            x[flip_mask] ^= 1  # flip bits

        return x

    def encode_systematic(self, message: np.ndarray) -> np.ndarray:
        """
        Very crude "encoding": pad/truncate the message to length n
        and then project it to the nearest codeword by decoding.

        This is NOT a proper encoder in the coding-theory sense,
        but is enough to create a codeword from an arbitrary bitstring.
        """
        msg = np.asarray(message, dtype=int) % 2

        if msg.shape[0] >= self.n:
            x0 = msg[:self.n]
        else:
            x0 = np.zeros(self.n, dtype=int)
            x0[:msg.shape[0]] = msg

        # Project to nearest codeword (in this crude sense)
        x_code = self.bit_flip_decode(x0, max_iter=100)
        return x_code


# ----------------------------------------------------------------------
# 2. VECTORIZED PANEL ECONOMETRIC MODEL
# ----------------------------------------------------------------------

class VectorizedPanelModel:
    """
    Vectorized panel econometric model:

    y_{i,t} = X_{i,t} beta + u_i + eps_{i,t}
    - Works on tensors (T, R, S, K) → flattened to (N_obs, K).
    - Simple FE-style: demeans by entity (region-sector) if requested.

    This is deliberately minimal. You can later slot in:
    - System-GMM
    - VECM
    - Quantum-inspired MC / Markov overlays
    """

    def __init__(self, use_fixed_effects: bool = True):
        self.beta: Optional[np.ndarray] = None
        self.use_fixed_effects = use_fixed_effects
        self.entity_ids: Optional[np.ndarray] = None  # For FE transforms
        self.resid_var: Optional[float] = None

    def _reshape_panel(self,
                       y: np.ndarray,
                       X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reshape from tensors to 2D forms.

        Parameters
        ----------
        y : np.ndarray
            Shape: (T, R, S)
        X : np.ndarray
            Shape: (T, R, S, K)

        Returns
        -------
        y_vec : (N, 1)
        X_mat : (N, K)
        entity_ids : (N,) unique entities (for FE)
        """
        if y.ndim != 3:
            raise ValueError("y must have shape (T, R, S)")
        if X.ndim != 4:
            raise ValueError("X must have shape (T, R, S, K)")

        T, R, S = y.shape
        _, _, _, K = X.shape

        # Flatten
        y_vec = y.reshape(T * R * S, 1)
        X_mat = X.reshape(T * R * S, K)

        # Entity index: region-sector combination
        r_idx, s_idx = np.meshgrid(np.arange(R), np.arange(S), indexing='ij')
        entity = (r_idx * S + s_idx).ravel()  # R*S unique entities
        entity_ids = np.tile(entity, T)       # repeated across time

        return y_vec, X_mat, entity_ids

    def _demean_by_entity(self,
                          y_vec: np.ndarray,
                          X_mat: np.ndarray,
                          entity_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Within transformation (demeaning) by entity.
        """
        N = y_vec.shape[0]
        y_demeaned = np.empty_like(y_vec)
        X_demeaned = np.empty_like(X_mat)

        # Vectorized approach using group means
        # 1) sort by entity
        sort_idx = np.argsort(entity_ids)
        y_sorted = y_vec[sort_idx]
        X_sorted = X_mat[sort_idx]
        ent_sorted = entity_ids[sort_idx]

        # We need group boundaries
        boundaries = np.concatenate([[0],
                                     np.flatnonzero(np.diff(ent_sorted)) + 1,
                                     [N]])
        # For each group:
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            length = end - start
            if length == 0:
                continue

            y_group = y_sorted[start:end]
            X_group = X_sorted[start:end]
            y_mean = np.mean(y_group, axis=0)
            X_mean = np.mean(X_group, axis=0)

            y_sorted[start:end] = y_group - y_mean
            X_sorted[start:end] = X_group - X_mean

        # 3) unsort back
        inv_idx = np.empty_like(sort_idx)
        inv_idx[sort_idx] = np.arange(N)

        y_demeaned = y_sorted[inv_idx]
        X_demeaned = X_sorted[inv_idx]

        return y_demeaned, X_demeaned

    def fit(self,
            y: np.ndarray,
            X: np.ndarray):
        """
        Fit the model via OLS, with optional FE transform.

        Parameters
        ----------
        y : np.ndarray
            Outcome tensor, shape: (T, R, S)
        X : np.ndarray
            Design tensor, shape: (T, R, S, K)
        """
        y_vec, X_mat, entity_ids = self._reshape_panel(y, X)
        self.entity_ids = entity_ids

        if self.use_fixed_effects:
            y_use, X_use = self._demean_by_entity(y_vec, X_mat, entity_ids)
        else:
            y_use, X_use = y_vec, X_mat

        # OLS: beta = (X'X)^(-1) X'y, use pseudo-inverse for stability
        XtX = X_use.T @ X_use
        Xty = X_use.T @ y_use
        self.beta = (np.linalg.pinv(XtX) @ Xty).flatten()  # (K,)

        # Residual variance (scalar)
        y_hat = X_use @ self.beta[:, None]
        resid = y_use - y_hat
        dof = X_use.shape[0] - X_use.shape[1]
        self.resid_var = float((resid.T @ resid).item() / dof)

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Predict outcome tensor given new regressors X_new.

        Parameters
        ----------
        X_new : np.ndarray
            Shape: (T, R, S, K)

        Returns
        -------
        y_hat : np.ndarray
            Shape: (T, R, S)
        """
        if self.beta is None:
            raise ValueError("Model not fitted.")

        T, R, S, K = X_new.shape
        X_mat = X_new.reshape(T * R * S, K)
        y_hat_vec = X_mat @ self.beta
        return y_hat_vec.reshape(T, R, S)

    def summary(self) -> Dict[str, np.ndarray]:
        """
        Simple summary of coefficients and residual variance.
        """
        return {
            "beta": self.beta,
            "resid_var": self.resid_var
        }


# ----------------------------------------------------------------------
# 3. QUANTUM POLICY SIMULATOR WRAPPED AROUND THE ECONOMETRIC CORE
# ----------------------------------------------------------------------

class QuantumPolicySimulator:
    """
    Combines PolicySpace (quantum state over policy configs) and
    VectorizedPanelModel (econometric outcome model).

    Workflow:
    1. Fit econometric model on historical data (y, X).
    2. Define how policy levers enter X (through a policy design function).
    3. Evolve quantum policy state psi via operators (policy shocks, regime shifts).
    4. At each time step, compute E[policy_features] and build X_policy(t).
    5. Use econometric model to predict outcomes under these policy expectations.
    """

    def __init__(self,
                 policy_space: PolicySpace,
                 panel_model: VectorizedPanelModel,
                 base_X: np.ndarray,
                 policy_sensitivity: np.ndarray):
        """
        Parameters
        ----------
        policy_space : PolicySpace
            Defined above.
        panel_model : VectorizedPanelModel
            Fitted econometric model.
        base_X : np.ndarray
            Baseline regressors (without policy terms), shape (T, R, S, K).
            K must match panel_model.beta dimension.

        policy_sensitivity : np.ndarray
            Tensor that maps policy features into regressors.
            Shape: (n_policy_features, K).
            Interpretation:
                For each policy feature p and regressor k,
                X_policy[..., k] += policy_feature[p] * policy_sensitivity[p, k]
        """
        self.policy_space = policy_space
        self.panel_model = panel_model
        self.base_X = base_X.copy()
        self.policy_sensitivity = policy_sensitivity

        self.T, self.R, self.S, self.K = base_X.shape
        self.n_policy_features = policy_sensitivity.shape[0]

        if self.policy_sensitivity.shape[1] != self.K:
            raise ValueError("policy_sensitivity must have shape (P, K) with K matching base_X.")

    def _build_policy_adjusted_X(self,
                                 policy_features: np.ndarray) -> np.ndarray:
        """
        Build regressors under a given policy feature vector.

        Parameters
        ----------
        policy_features : np.ndarray
            Shape: (P,)

        Returns
        -------
        X_adj : np.ndarray
            Shape: (T, R, S, K)
        """
        # broadcast policy_features (P,) via einsum onto sensitivity (P, K)
        # giving delta_K (K,)
        delta_k = np.einsum("p,pk->k", policy_features, self.policy_sensitivity)
        # Now broadcast delta_k over (T, R, S)
        X_adj = self.base_X + delta_k.reshape(1, 1, 1, self.K)
        return X_adj

    def simulate_horizon(self,
                         U_sequence: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Simulate over a discrete horizon, applying a sequence of
        policy operators U_t to psi.

        Parameters
        ----------
        U_sequence : list of np.ndarray
            Each U_t is a (dim, dim) operator on policy space at step t.

        Returns
        -------
        dict
            {
              "policy_features": array of shape (H, P),
              "y_hat": array of shape (H, T, R, S),
              "psi_trajectories": array of shape (H, dim)
            }
            where H = len(U_sequence).
        """
        H = len(U_sequence)
        dim = self.policy_space.dim
        P = self.n_policy_features

        policy_features_hist = np.zeros((H, P))
        y_hat_hist = np.zeros((H, self.T, self.R, self.S))
        psi_hist = np.zeros((H, dim), dtype=np.complex128)

        for t in range(H):
            # Apply policy operator for step t
            self.policy_space.apply_unitary(U_sequence[t])

            # Store psi
            psi_hist[t, :] = self.policy_space.psi

            # Compute expected policy features
            pf = self.policy_space.expectation_policy_features()
            policy_features_hist[t, :] = pf

            # Build adjusted X and predict outcomes
            X_adj = self._build_policy_adjusted_X(pf)
            y_hat = self.panel_model.predict(X_adj)
            y_hat_hist[t, :, :, :] = y_hat

        return {
            "policy_features": policy_features_hist,
            "y_hat": y_hat_hist,
            "psi_trajectories": psi_hist
        }

    def evaluate_policy_config_cost(
        self,
        config_id: int,
        aggregator: Callable[[np.ndarray], float]
    ) -> float:
        """Evaluate scalar cost for a single policy configuration."""
        pf = self.policy_space.config_id_to_feature_vector(config_id)
        X_adj = self._build_policy_adjusted_X(pf)
        y_hat = self.panel_model.predict(X_adj)
        return float(aggregator(y_hat))

    def build_qubo_over_policy_configs(
        self,
        aggregator: Callable[[np.ndarray], float],
        penalty_lambda: float = 10.0,
        max_configs: Optional[int] = None
    ) -> QUBOModel:
        """Build a QUBO where each binary variable corresponds to a policy config."""
        n_configs = self.policy_space.config_count()
        if max_configs is not None:
            n_configs = min(n_configs, max_configs)

        utilities = np.zeros(n_configs)
        for cid in range(n_configs):
            utilities[cid] = self.evaluate_policy_config_cost(cid, aggregator)

        qubo = build_policy_config_qubo(utilities, penalty_lambda=penalty_lambda)
        return qubo

    def optimise_policy_via_qubo(
        self,
        aggregator: Callable[[np.ndarray], float],
        penalty_lambda: float = 10.0,
        max_configs: Optional[int] = None
    ) -> Dict[str, object]:
        """Optimise over policy configs using the QUBO formulation."""
        qubo = self.build_qubo_over_policy_configs(
            aggregator=aggregator,
            penalty_lambda=penalty_lambda,
            max_configs=max_configs,
        )

        z_star, E_star = qubo.brute_force_solve()
        chosen_idx = int(np.argmax(z_star))

        pf = self.policy_space.config_id_to_feature_vector(chosen_idx)
        X_adj = self._build_policy_adjusted_X(pf)
        y_hat = self.panel_model.predict(X_adj)

        return {
            "qubo": qubo,
            "z_star": z_star,
            "E_star": E_star,
            "chosen_config_id": chosen_idx,
            "chosen_policy_features": pf,
            "y_hat": y_hat,
        }

# ----------------------------------------------------------------------
# 4. SYNTHETIC DATA GENERATOR FOR TESTING (IRISH-STYLE PANEL)
# ----------------------------------------------------------------------

def generate_synthetic_irish_agri_panel(T: int = 20,
                                        regions: List[str] = None,
                                        sectors: List[str] = None,
                                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate toy panel data representing Irish agri outcomes.

    y   : farm income or yield index
    X   : regressors [const, rainfall, temp, lagged_y, carbon_price]

    This is just to *run* the system; replace with real CSO/DAFM/Eurostat data.
    """
    rng = np.random.default_rng(seed)

    if regions is None:
        regions = ["Cork", "Kerry", "Galway", "Tipperary"]
    if sectors is None:
        sectors = ["Dairy", "Beef", "Tillage"]

    R = len(regions)
    S = len(sectors)
    K = 5  # [const, rainfall, temp, lagged_y, carbon_price]

    # Base climatology and carbon price patterns
    rainfall = rng.normal(loc=1000, scale=150, size=(T, R, S))   # mm/year
    temp = rng.normal(loc=10, scale=1.0, size=(T, R, S))         # deg C
    carbon_price = np.linspace(20, 80, T).reshape(T, 1, 1)       # €/tonne
    carbon_price = carbon_price * np.ones((1, R, S))

    y = np.zeros((T, R, S))
    # initialize base_y
    base_y = rng.normal(loc=100, scale=10, size=(R, S))

    for t in range(T):
        if t == 0:
            lag_y = base_y
        else:
            lag_y = y[t - 1]

        # True DGP (toy):
        # y = 50 + 0.05*rainfall + 2*temp + 0.6*lag_y - 0.3*carbon_price + noise
        noise = rng.normal(loc=0, scale=5.0, size=(R, S))
        y[t] = (50
                + 0.05 * rainfall[t]
                + 2.0 * temp[t]
                + 0.6 * lag_y
                - 0.3 * carbon_price[t]
                + noise)

    # Build X tensor
    X = np.zeros((T, R, S, K))
    X[..., 0] = 1.0                      # const
    X[..., 1] = rainfall
    X[..., 2] = temp
    # lagged y (forward-fill last period)
    for t in range(T):
        if t == 0:
            X[t, ..., 3] = base_y
        else:
            X[t, ..., 3] = y[t - 1]
    X[..., 4] = carbon_price

    meta = {
        "regions": regions,
        "sectors": sectors,
        "regressor_names": ["const", "rainfall", "temp", "lag_y", "carbon_price"]
    }

    return y, X, meta

def build_policy_config_qubo(
    utilities: np.ndarray,
    penalty_lambda: float = 10.0
) -> QUBOModel:
    """
    Build a QUBO where each binary variable z_j selects policy config j.

    utilities[j] = scalar cost of config j  (lower = better).

    Objective (conceptually):
        minimize sum_j utilities[j] * z_j
        subject to sum_j z_j = 1

    We encode the constraint with a quadratic penalty:
        penalty_lambda * (sum_j z_j - 1)^2

    The resulting E(z) can be written as z^T Q z, so we derive Q.
    """
    utilities = np.asarray(utilities, dtype=float)
    n = utilities.shape[0]

    # Linear part: utilities[j] * z_j
    # Quadratic penalty: lambda * (sum z_j - 1)^2
    # = lambda * (sum z_j^2 + 2 sum_{i<j} z_i z_j - 2 sum_j z_j + 1)
    # Note: z_j^2 = z_j for binary variables.

    # Start with all zeros
    Q = np.zeros((n, n), dtype=float)

    # Diagonal: utilities + penalty_lambda * (1 - 2)
    # from z_j term and constraint expansion
    for j in range(n):
        # From utilities: linear term mapped to Q_jj
        Q[j, j] += utilities[j]

        # From penalty: z_j^2 term -> lambda
        # and linear part -2*lambda*z_j is also
        # put on diagonal (since diag corresponds to linear in QUBO)
        Q[j, j] += penalty_lambda * (1.0 - 2.0)

    # Off-diagonal: 2 * penalty_lambda for i<j
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += 2.0 * penalty_lambda

    return QUBOModel(Q)


# ----------------------------------------------------------------------
# 5. EXAMPLE WIRING – BUILD FULL SYSTEM
# ----------------------------------------------------------------------

def build_example_system():
    """
    End-to-end example:
    - Build synthetic Irish panel
    - Define policy levers (subsidy, carbon tax, R&D)
    - Construct policy space
    - Fit econometric model
    - Build quantum policy simulator
    - Simulate horizon H with random operators
    """
    # 1) Synthetic data
    y, X, meta = generate_synthetic_irish_agri_panel(T=20)

    # 2) Policy levers – you can make this more granular later
    levers = {
        "fertiliser_subsidy": [0.0, 0.5, 1.0],   # fraction of baseline cost subsidised
        "carbon_tax": [0.0, 50.0, 100.0],        # €/tonne
        "R_and_D_support": [0.0, 0.3, 0.6]       # share of farm revenue reinvested
    }

    policy_space = PolicySpace(levers)

    # 3) Fit panel model
    panel_model = VectorizedPanelModel(use_fixed_effects=True)
    panel_model.fit(y, X)
    print("Fitted coefficients:", panel_model.summary())

    # 4) Policy sensitivity mapping
    #    Map each policy feature to how it perturbs regressors.
    #    Here, we define a simple mapping:
    #    - fertiliser_subsidy -> effectively reduces carbon_price impact
    #    - carbon_tax         -> increases regressor "carbon_price"
    #    - R&D_support        -> increases lagged_y effect (productivity gains)
    K = X.shape[3]
    P = len(levers)

    policy_sensitivity = np.zeros((P, K))
    regressor_names = meta["regressor_names"]
    idx_const = regressor_names.index("const")
    idx_lag_y = regressor_names.index("lag_y")
    idx_carbon = regressor_names.index("carbon_price")

    # fertiliser_subsidy: reduce effective carbon_price
    policy_sensitivity[0, idx_carbon] = -0.5  # negative: subsidy offsets carbon burden

    # carbon_tax: increase carbon_price
    policy_sensitivity[1, idx_carbon] = +1.0

    # R_and_D_support: enhance lagged_y effect (productivity)
    policy_sensitivity[2, idx_lag_y] = +0.3

    # 5) Build simulator
    simulator = QuantumPolicySimulator(
        policy_space=policy_space,
        panel_model=panel_model,
        base_X=X,
        policy_sensitivity=policy_sensitivity
    )

    # 6) Build a sequence of random "policy regime" operators
    H = 5  # policy horizon in steps (can be years)
    U_seq = []
    for t in range(H):
        # Could also use kronecker_policy_operator with local_mats
        U_t = policy_space.random_unitary(seed=100 + t)
        U_seq.append(U_t)

    # 7) Simulate horizon
    results = simulator.simulate_horizon(U_seq)

    # Unpack
    policy_features_hist = results["policy_features"]
    y_hat_hist = results["y_hat"]

    print("\nPolicy features over horizon (E[lever levels]):")
    print(pd.DataFrame(policy_features_hist, columns=policy_space.lever_names))

    print("\nExample: Predicted y for step 0, first time, region, sector:")
    print("y_hat[0, 0, :, :]:\n", y_hat_hist[0, 0])

    # QUBO demo: choose a policy configuration that maximises average predicted outcome
    aggregator = lambda y_hat: -float(np.mean(y_hat))
    qubo_result = simulator.optimise_policy_via_qubo(
        aggregator=aggregator,
        penalty_lambda=10.0,
        max_configs=None,
    )

    print("\nQUBO optimisation result:")
    print("Chosen config id:", qubo_result["chosen_config_id"])
    print("Chosen policy features:",
          dict(zip(policy_space.lever_names, qubo_result["chosen_policy_features"])))

    return {
        "policy_space": policy_space,
        "panel_model": panel_model,
        "simulator": simulator,
        "results": results,
        "meta": meta,
        "qubo_result": qubo_result
    }


if __name__ == "__main__":
    # Run the example system when executed as a script
    system_objects = build_example_system()
