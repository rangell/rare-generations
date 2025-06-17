import numpy as np
import torch
from typing import Callable, List, Optional, Tuple, Union


class FKSteering:

    def __init__(
        self,
        device,
        r_fn,
        potential_type,
        max_seq_len,
        num_particles,
        resample_start,
        resample_end,
        resample_interval,
        lmbda,
    ):
        self.device = device
        self.r_fn = r_fn
        self.lmbda = lmbda

        self.num_particles = num_particles

        self.resample_start = resample_start
        self.resample_end = resample_end
        self.resample_interval = resample_interval
        self.max_seq_len = max_seq_len

        self.resampling_arr = np.arange(
            resample_start, resample_end + 1, resample_interval
        )
        self.resampling_arr = torch.cat(
            [self.resampling_arr, torch.tensor([max_seq_len])]
        )

        self.potential_type = potential_type
        assert potential_type in ["r_fn", "max", "diff", "bon"], potential_type

        self.r_values = [torch.zeros(num_particles, device=device)]
        self.potential_values = []

    def resampling_fn(self, w):
        """
        Resampling function that returns indices based on the importance weights.

        Input:
        w: unnormalized importance weights, shape (N,)

        Output:
        indices: indices for resampling, shape (N,)
        """
        num_particles = w.shape[0]
        ess = 1.0 / np.sum(np.square(w))

        if ess < 0.5 * num_particles:
            indices = np.random.choice(
                num_particles, size=num_particles, p=w / np.sum(w)
            )
        else:
            indices = np.arange(num_particles)
        indices = torch.tensor(indices, dtype=torch.long, device=self.device)

        return indices

    def compute_potential(self, sample_idx, sequence, rs_candidates):
        if self.potential_type == "r_fn":
            raise NotImplementedError("r_fn potential type not implemented")
        elif self.potential_type == "max":
            raise NotImplementedError("Max potential type not implemented")
        elif self.potential_type == "diff":
            rs_old = self.r_values[-1]

            return torch.exp(self.lmbda * (rs_candidates - rs_old))
        elif self.potential_type == "bon":
            if sample_idx == 0:
                return torch.ones_like(rs_candidates)

    def update_history(self, indices, r_values, potential_values):
        for past_idx in range(len(self.potential_values)):
            r_values[past_idx] = r_values[past_idx][indices]
            potential_values[past_idx] = potential_values[past_idx][indices]

        return r_values, potential_values

    def __call__(self, sample_idx, sequences, importance_weights):
        if sample_idx not in self.resampling_arr:
            return sequences

        rs_candidates = self.r_fn(sequences)
        assert rs_candidates.shape == (self.num_particles,), rs_candidates.shape

        potential_values = self.compute_potential(sample_idx, sequences, rs_candidates)
        assert potential_values.shape == (self.num_particles,), potential_values.shape

        normalized_potential = importance_weights * potential_values
        normalized_potential /= torch.sum(normalized_potential)
        assert normalized_potential.shape == (
            self.num_particles,
        ), normalized_potential.shape

        indices = self.resampling_fn(normalized_potential)
        resampled_sequence = sequences[indices]

        assert resampled_sequence.shape == (
            self.num_particles,
            self.max_seq_len,
        ), resampled_sequence.shape

        # Update r_values and potential_values to new indices
        r_values, potential_values = self.update_history(
            indices, r_values=self.r_values, potential_values=potential_values
        )

        self.r_values.append(r_values[indices])
        self.potential_values.append(potential_values[indices])

        return resampled_sequence

    def compute_fk_estimate(self, test_function_values):
        assert (
            self.potential_type == "diff"
        ), "FK estimate only available for 'diff' potential type"

        r_0 = self.r_values[0]
        r_T = self.r_values[-1]

        assert r_0.shape == r_T.shape == (self.num_particles,)

        product_of_potentials = torch.exp(self.lmbda * (r_T - r_0))
        assert product_of_potentials.shape == (self.num_particles,)
        
        inv_potential = torch.exp(-self.lmbda * (r_T - r_0)) 
        assert inv_potential.shape == (self.num_particles,)

        Z = torch.mean(product_of_potentials)
        assert Z > 0, "Z must be positive for FK estimate"
        
        estimate = Z * (test_function_values * inv_potential).mean().item()
        return estimate


# class SMC:
#     """
#     Sequential Monte Carlo implementation with multinomial resampling.

#     This class implements the Sequential Monte Carlo (SMC) algorithm with
#     multinomial resampling, verification capabilities, and tracking of
#     importance weights.
#     """

#     def __init__(
#         self,
#         num_particles: int,
#         verifier: Callable,
#         current_seqs: Optional[List] = None,
#         importance_weights: Optional[torch.Tensor] = None,
#         device: torch.device = torch.device("cpu"),
#     ):
#         """
#         Initialize the SMC algorithm.

#         Args:
#             num_particles: Number of particles to use
#             verifier: Function that evaluates if a particle satisfies a condition
#             current_seqs: Initial sequences for particles
#             importance_weights: Initial importance weights (in log space)
#             device: Device to run computations on
#         """
#         self.num_particles = num_particles
#         self.verifier = verifier
#         self.device = device

#         # Initialize log weights
#         if importance_weights is None:
#             self.log_weights = torch.zeros(num_particles, device=device)
#         else:
#             self.log_weights = importance_weights

#         # Initialize normalized weights
#         self._normalize_weights()

#         # Initialize current sequences
#         self.current_seqs = current_seqs if current_seqs is not None else []

#         # Tracking variables
#         self.resampling_count = 0
#         self.history = []

#     def _normalize_weights(self):
#         """Normalize weights using log-sum-exp trick for numerical stability."""
#         if len(self.log_weights) == 0:
#             self.normalized_weights = torch.tensor([], device=self.device)
#             return

#         max_log_weight = torch.max(self.log_weights)
#         self.normalized_weights = torch.exp(self.log_weights - max_log_weight)
#         sum_weights = torch.sum(self.normalized_weights)

#         if sum_weights > 0:
#             self.normalized_weights /= sum_weights
#         else:
#             # Default to uniform weights if all weights are effectively zero
#             self.normalized_weights = (
#                 torch.ones_like(self.log_weights) / len(self.log_weights)
#             )

#     def set_current_seqs(self, seqs: List):
#         """
#         Set the current sequences for all particles.

#         Args:
#             seqs: List of sequences, should match num_particles in length
#         """
#         if len(seqs) != self.num_particles:
#             raise ValueError(
#                 f"Expected {self.num_particles} sequences, got {len(seqs)}"
#             )
#         self.current_seqs = seqs

#     def update_weights(self, importance_weights: torch.Tensor):
#         """
#         Update particle weights using importance weights.

#         Args:
#             importance_weights: Log importance weights for each particle
#         """
#         if len(importance_weights) != self.num_particles:
#             raise ValueError(
#                 f"Expected {self.num_particles} weights, got {len(importance_weights)}"
#             )

#         # Update log weights
#         self.log_weights += importance_weights

#         # Normalize weights
#         self._normalize_weights()

#         return self.normalized_weights

#     def calculate_ess(self) -> float:
#         """
#         Calculate effective sample size.

#         Returns:
#             Normalized effective sample size (0 to 1)
#         """
#         if len(self.normalized_weights) == 0:
#             return 0.0

#         ess = 1.0 / torch.sum(self.normalized_weights ** 2)
#         normalized_ess = ess / self.num_particles

#         return normalized_ess.item()

#     def resample_multinomial(self) -> Tuple[torch.Tensor, List]:
#         """
#         Perform multinomial resampling based on current weights.

#         Returns:
#             Tuple containing:
#                 - ancestor_indices: Indices of selected particles
#                 - resampled_seqs: The resampled sequences
#         """
#         if not self.current_seqs:
#             raise ValueError("No current sequences available for resampling")

#         # Multinomial resampling
#         ancestor_indices = torch.multinomial(
#             self.normalized_weights, self.num_particles, replacement=True
#         )

#         # Resample sequences
#         resampled_seqs = [self.current_seqs[idx] for idx in ancestor_indices]
#         self.current_seqs = resampled_seqs

#         # Reset weights to uniform
#         self.log_weights = torch.zeros(self.num_particles, device=self.device)
#         self.normalized_weights = (
#             torch.ones(self.num_particles, device=self.device) / self.num_particles
#         )

#         # Update tracking
#         self.resampling_count += 1
#         self.history.append(
#             {"step": self.resampling_count, "ess_before": self.calculate_ess()}
#         )

#         return ancestor_indices, resampled_seqs

#     def resample_if_needed(self, threshold: float = 0.5) -> Optional[Tuple[torch.Tensor, List]]:
#         """
#         Resample particles if effective sample size falls below threshold.

#         Args:
#             threshold: Threshold for normalized effective sample size (0 to 1)

#         Returns:
#             Resampling results if performed, otherwise None
#         """
#         normalized_ess = self.calculate_ess()

#         if normalized_ess < threshold:
#             return self.resample_multinomial()

#         return None

#     def verify_particles(self) -> torch.Tensor:
#         """
#         Apply verifier function to all current particles.

#         Returns:
#             Boolean tensor indicating which particles satisfy verification
#         """
#         if not self.current_seqs:
#             return torch.tensor([], dtype=torch.bool, device=self.device)

#         verification_results = []
#         for seq in self.current_seqs:
#             verification_results.append(self.verifier(seq))

#         return torch.tensor(verification_results, dtype=torch.bool, device=self.device)

#     def get_verified_particles(self) -> Tuple[List, torch.Tensor]:
#         """
#         Get particles that pass verification along with their weights.

#         Returns:
#             Tuple of verified particles and their normalized weights
#         """
#         verified_mask = self.verify_particles()

#         if torch.sum(verified_mask) == 0:
#             return [], torch.tensor([], device=self.device)

#         verified_particles = [
#             self.current_seqs[i] for i in range(len(self.current_seqs)) if verified_mask[i]
#         ]
#         verified_weights = self.normalized_weights[verified_mask]

#         # Renormalize weights
#         if len(verified_weights) > 0 and torch.sum(verified_weights) > 0:
#             verified_weights = verified_weights / torch.sum(verified_weights)

#         return verified_particles, verified_weights

#     def get_best_particle(self) -> Optional[Tuple[any, float]]:
#         """
#         Get particle with highest weight.

#         Returns:
#             Tuple of (best particle, weight) or None if no particles
#         """
#         if not self.current_seqs or len(self.normalized_weights) == 0:
#             return None

#         best_idx = torch.argmax(self.normalized_weights).item()
#         return self.current_seqs[best_idx], self.normalized_weights[best_idx].item()

#     def get_results(self) -> dict:
#         """
#         Get comprehensive results of the SMC process.

#         Returns:
#             Dictionary with results including particles, weights, and statistics
#         """
#         verified_particles, verified_weights = self.get_verified_particles()
#         verified_mask = self.verify_particles()

#         best_result = self.get_best_particle()
#         best_particle = best_result[0] if best_result else None
#         best_weight = best_result[1] if best_result else 0.0

#         return {
#             "best_particle": best_particle,
#             "best_particle_weight": best_weight,
#             "verified_particles": verified_particles,
#             "verified_weights": verified_weights,
#             "verification_rate": torch.mean(verified_mask.float()).item()
#             if len(verified_mask) > 0
#             else 0.0,
#             "resampling_count": self.resampling_count,
#             "ess": self.calculate_ess(),
#             "all_particles": self.current_seqs,
#             "all_weights": self.normalized_weights,
#             "history": self.history,
#         }
