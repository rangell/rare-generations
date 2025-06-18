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
        use_smc,        
    ):  
        self.use_smc = use_smc
        
        self.device = device
        self.r_fn = r_fn
        self.lmbda = lmbda

        self.num_particles = num_particles

        self.resample_start = resample_start
        self.resample_end = resample_end
        self.resample_interval = resample_interval
        self.max_seq_len = max_seq_len

        self.resampling_arr = torch.arange(
            resample_start, resample_end + 1, resample_interval
        )
        self.resampling_arr = torch.cat(
            [self.resampling_arr, torch.tensor([max_seq_len])]
        )

        self.potential_type = potential_type
        assert potential_type in ["r_fn", "max", "diff", "bon"], potential_type

        self.arr_r_values = [torch.zeros(num_particles, device=device)]
        self.arr_potential_values = []

    def resampling_fn(self, w):
        """
        Resampling function that returns indices based on the importance weights.

        Input:
        w: unnormalized importance weights, shape (N,)

        Output:
        indices: indices for resampling, shape (N,)
        """
        num_particles = w.shape[0]

        # Normalize the weights
        normalized_w = w / torch.sum(w)
        ess = 1.0 / torch.sum(torch.pow(normalized_w, 2)).item()

        if ess < 0.5 * num_particles:
            print("Resampling triggered due to low ESS:", ess)
            indices = np.random.choice(
                num_particles, size=num_particles, p=normalized_w.cpu().numpy()
            )
        else:
            indices = np.arange(num_particles)
        # indices = torch.tensor(indices, dtype=torch.long, device=self.device)

        return indices

    def compute_potential(self, sample_idx, sequence, rs_candidates):
        if self.potential_type == "r_fn":
            raise NotImplementedError("r_fn potential type not implemented")
        elif self.potential_type == "max":
            raise NotImplementedError("Max potential type not implemented")
        elif self.potential_type == "diff":
            rs_old = self.arr_r_values[-1]
            return torch.exp(self.lmbda * (rs_candidates - rs_old))
        
        elif self.potential_type == "bon":
            if sample_idx == 0:
                return torch.ones_like(rs_candidates)
        else:
            raise ValueError(f"Unknown potential type: {self.potential_type}")

    def update_history(self, indices, arr_r_values, arr_potential_values):
        for past_idx in range(len(arr_potential_values)):
            arr_r_values[past_idx] = arr_r_values[past_idx][indices]
            arr_potential_values[past_idx] = arr_potential_values[past_idx][indices]

        return arr_r_values, arr_potential_values

    def __call__(self, sample_idx, sequences, importance_weights):
        # print("resampling call")
        if sample_idx not in self.resampling_arr or not self.use_smc:
            return sequences, torch.arange(self.num_particles, device=self.device)

        rs_candidates = self.r_fn(sequences)
        assert rs_candidates.shape == (self.num_particles,), rs_candidates.shape

        potential_values = self.compute_potential(sample_idx, sequences, rs_candidates)
        potential_values = potential_values * importance_weights.view(self.num_particles)
        assert potential_values.shape == (self.num_particles,), potential_values.shape
      
        normalized_potential = potential_values / torch.sum(potential_values)
        assert normalized_potential.shape == (self.num_particles,), (
            normalized_potential.shape,
            importance_weights.shape,
            potential_values.shape,
        )

        indices = self.resampling_fn(normalized_potential)

        num_particles, seq_len = sequences.shape
        resampled_sequence = sequences[indices]

        assert resampled_sequence.shape == (
            self.num_particles,
            seq_len,
        ), (resampled_sequence.shape, self.num_particles, seq_len)

        # Update r_values and potential_values to new indices
        arr_r_values, arr_potential_values = self.update_history(
            indices,
            arr_r_values=self.arr_r_values,
            arr_potential_values=self.arr_potential_values,
        )
        
        self.arr_r_values = arr_r_values
        self.arr_potential_values = arr_potential_values
        
        self.arr_r_values.append(rs_candidates[indices])
        self.arr_potential_values.append(potential_values[indices])

        return resampled_sequence, indices

    def compute_fk_estimate(self, test_function_values):
        assert (
            self.potential_type == "diff"
        ), "FK estimate only available for 'diff' potential type"

        r_0 = self.arr_r_values[0]
        r_T = self.arr_r_values[-1]

        assert r_0.shape == r_T.shape == (self.num_particles,)

        # product_of_potentials = torch.exp(self.lmbda * (r_T - r_0)) 
        product_of_potentials = torch.prod(
            torch.stack(self.arr_potential_values, dim=0), dim=0
        )
        assert product_of_potentials.shape == (self.num_particles,)

        inv_potential = 1. / product_of_potentials
        assert inv_potential.shape == (self.num_particles,)

        Z = torch.mean(product_of_potentials)
        assert Z > 0, "Z must be positive for FK estimate"
        
        print("Z:", Z.item())
        print('inv_potential:', inv_potential.mean().item())
        print('product_of_potentials:', product_of_potentials.mean().item())

        estimate = Z * (test_function_values * inv_potential).mean().item()
        return estimate
