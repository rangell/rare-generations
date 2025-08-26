from est_unsafe_to_unsafe import get_exp_args


def define_grid():

    grid = {
        "ablation_intensity": [0.1, 0.3, 0.5, 0.7],
        "proposal_model": ["toxic_model"],
        "use_smc": [True, False],
        "model_idx": [
            0,  # meta-llama/Llama-3.2-1B-Instruct
            1,  # meta-llama/meta-llama-3-8b-instruct
            2,  # google/gemma-2-9b-it
            3,  # google/gemma-2-2b-it
            4,  # GraySwanAI/Llama-3-8B-Instruct-RR
        ],
        "cheap_judge": [True],
    }

    return grid

def generate_args(exp_args, grid, grid_index):
    hp_grid = []
    # Separate list parameters from scalar parameters
    list_params = {k: v for k, v in grid.items() if isinstance(v, list)}
    scalar_params = {k: v for k, v in grid.items() if not isinstance(v, list)}

    param_names = list(list_params.keys())
    param_values = list(list_params.values())

    # Generate all combinations using itertools.product
    all_combinations = list(itertools.product(*param_values))

    # Filter invalid combinations
    hp_grid = []
    for combo in tqdm(all_combinations):
        # Create parameter dictionary for this combination
        params = dict(zip(param_names, combo))
        params.update(scalar_params)

        if params not in hp_grid:
            hp_grid.append(params)

    new_args = hp_grid[grid_index]

    print(f"Selecting {grid_index} out of {len(hp_grid)} configurations")
    # import pdb; pdb.set_trace()

    # exit()

    # Update args with the new configuration
    for key, value in new_args.items():
        setattr(exp_args, key, value)

    print(exp_args)
    return exp_args


def main():
    exp_args = get_exp_args()


if __name__ == "__main__":
    grid = define_grid()
    exp_args = generate_args(exp_args, grid, 0)