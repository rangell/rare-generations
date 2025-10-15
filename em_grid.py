import argparse
import yaml
from pathlib import Path
import os
import itertools
from tqdm import tqdm

from est_safe_to_unsafe import main as launch_experiment


def create_grid():
    grid = {
        "output_dir": "grid_search_safe_to_unsafe",  # TODO
        "use_smc": [False],
        "model_name": ["meta-llama/Llama-3.1-8B-Instruct"],
        "num_particles": [200],  # [100, 500],
        "proposal_idx_switch": [10, 15, 20],
        "lora_percent": [0.4, 0.5, 0.6],
        "proposal_bias": [0.4, 0.5, 0.6],
        "model_diff_amp_factor": [0.0],
        "adapter_name": [
            "ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports",
            "ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice",
            "ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice",
        ],
    }

    return grid


def generate_args(config_path, index):
    grid_dict = create_grid()

    hp_grid = []

    # Separate list parameters from scalar parameters
    list_params = {k: v for k, v in grid_dict.items() if isinstance(v, list)}
    scalar_params = {k: v for k, v in grid_dict.items() if not isinstance(v, list)}

    param_names = list(list_params.keys())
    param_values = list(list_params.values())

    # Generate all combinations using itertools.product
    all_combinations = list(itertools.product(*param_values))

    for combo in tqdm(all_combinations):
        # Create parameter dictionary for this combination
        params = dict(zip(param_names, combo))
        params.update(scalar_params)

        if params not in hp_grid:
            hp_grid.append(params)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = argparse.Namespace(**config)
    print(f"len(hp_grid): {len(hp_grid)}")

    new_args = hp_grid[index]

    print(f"Selecting {index + 1} out of {len(hp_grid)} configurations")

    # Update args with the new configuration
    for key, value in new_args.items():
        setattr(config, key, value)

    print(config)

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.yaml")
    parser.add_argument("--grid_index", type=int, required=True)
    args = parser.parse_args()

    config = generate_args(args.config_path, args.grid_index)

    launch_experiment(config)
