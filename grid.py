import argparse
import yaml
from pathlib import Path
import os
import itertools
from tqdm import tqdm

from est_unsafe_to_unsafe import main as launch_experiment

def create_grid():
    grid = {
    "output_dir": 'model_outputs',  # TODO 
    "experiment_identifier": 'grid_search_unsafe_to_unsafe',  # TODO experiment name
     "model_name": [],
     "num_particles": [],
     "proposal_model": [],
     "ablation_intensity": [],
     "use_smc": [True, False],
     "proposal_idx_switch": [],
     "proposal_bias": [],
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
    
    import pdb; pdb.set_trace()
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

    print(f"Selecting {index+1} out of {len(hp_grid)} configurations")
    # import pdb; pdb.set_trace()

    # exit()

    # Update args with the new configuration
    for key, value in new_args.items():
        setattr(args, key, value)

    print(args)

    return args
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.yaml")
    parser.add_argument("--grid_index", type=int, required=True)
    args = parser.parse_args()
    
    config = generate_args(args.config_path, args.grid_index)

    import pdb; pdb.set_trace()
    launch_experiment(config)
