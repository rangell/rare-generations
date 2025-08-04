
import argparse
from peft import PeftModel

from generate_and_judge import *


def get_em_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--num_particle_arr",
        type=int,
        nargs="+",
        default=[500, 500, 500],
        help="Number of particles for each generation step.",
    )
    parser.add_argument(
        "--proposal_model",
        type=str,
        default="toxic_model",
        choices=["base", "toxic_model"],
        help="Model to use for proposal generation.",
    )
    parser.add_argument(
        "--use_smc",
        action="store_true",
        help="Whether to use Sequential Monte Carlo (SMC) for generation.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="which model to score.",
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default="ModelOrganismsForEM/Llama-3.2-1B-Instruct_bad-medical-advice",
        help="adapter to use for the proposal model",
    )
    parser.add_argument(
        "--save_output",
        action="store_true",
        help="Whether to store the outputs in a log file",
    )

    parser.add_argument(
        "--smc_verbose", action="store_true", help="Whether to print SMC logs"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model_outputs",
        help="Store args, output probabilities",
    )

    args = parser.parse_args()

    smc_args = dict(
        decoding="sample",
        resample_start=10 if args.proposal_model == "base" else 10,
        resample_end=args.max_new_tokens - 10,
        resample_interval=10 if args.proposal_model == "base" else 20,
        adaptive_resampling=True,
        lmbda=10 if args.proposal_model == "base" else 10,
        base_inv_temperature=1.0,
        adaptive_resampling_threshold=0.5 if args.proposal_model == "base" else 0.5,
        proposal_inv_temperature=1.0,
        proposal_model_switch_idx=None,
    )

    metadata = {**smc_args, **vars(args)}

    combined_args = argparse.Namespace(**metadata)

    for key, value in metadata.items():
        print(f"{key}: {value}")

    if combined_args.save_output:
        output_dir = args.output_dir
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(output_dir) / args.model_name.split("/")[1] / timestamp
        try:
            os.makedirs(output_dir)
        except FileExistsError:
            print(f"Output directory {output_dir} already exists.")
            # wait for a random time to avoid overwriting
            wait_time = np.random.randint(1, 200)
            print(f"Waiting for {wait_time} seconds before proceeding...")
            time.sleep(wait_time)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(output_dir) / args.dataset / timestamp
            os.makedirs(output_dir, exist_ok=False)
        metadata["timestamp"] = timestamp

        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Experiment metadata saved to {metadata_file}")
    else:
        output_dir = None

    return combined_args, output_dir


if __name__ == "__main__":
    args, output_dir = get_em_args()
    seed_everything(args.seed)

    model_name_or_path = args.model_name
    refusal_direction_path = (
        f"refusal_direction/pipeline/runs/{args.model_name.split('/')[1]}/"
    )

    # Load model and tokenizer from huggingface
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)

    # Create the forward wrappers
    if args.proposal_model == "toxic_model":
        # load the adapter for the proposal model
        model = PeftModel.from_pretrained(
            model,
            args.adapter_name,
            device_map="auto",
            adapter_name="default",
        )
        base_forward = create_model_wrapper(
            model=model, tokenizer=tokenizer, enable_adapter=False
        )
        proposal_forward = create_model_wrapper(
            model=model, tokenizer=tokenizer, enable_adapter=True
        )
    else:
        base_forward = create_model_wrapper(
            model=model, tokenizer=tokenizer, enable_adapter=False
        )
        proposal_forward = base_forward

    mc_dataset = load_dataset(
        "json",
        data_files=f"monte_carlo_estimates/results/emergent_misalignment/{args.model_name.split('/')[1]}-first_plot_questions-mc_est.json",
    )["train"]

    max_new_tokens = args.max_new_tokens  # 150
    model_logs = {}
    for example_idx, example in enumerate(mc_dataset):
        if example["harm_mean"] == 0.0:
            continue

        example_model_logs = {}
        forbidden_prompt = example["forbidden_prompt"]
        messages = [
            {
                "role": "user",
                "content": forbidden_prompt,
            },
        ]
        arr_harmfulness_scores = []

        example_model_logs["Forbidden prompt"] = forbidden_prompt
        example_model_logs["MC mean"] = example["harm_mean"]
        example_model_logs["MC var"] = example["harm_var"]

        for num_particles_idx, num_particles in enumerate(args.num_particle_arr):
            convos = tokenizer.apply_chat_template(
                [messages for _ in range(num_particles)],
                tokenize=False,
                add_generation_prompt=True,
            )

            # Prepare the inputs
            inputs = tokenizer(convos, return_tensors="pt", padding=True)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            input_ids = inputs["input_ids"][:, 1:]
            attention_mask = inputs["attention_mask"][:, 1:]

            # reward function for SMC
            r_fn = create_reward_fn(
                tokenizer=tokenizer,
                forbidden_prompt=forbidden_prompt,
                prompt_len=input_ids.shape[1],
                batch_size=128,
            )

            # Clear cache to avoid OOM errors
            gc.collect()
            torch.cuda.empty_cache()

            # move model to eval mode
            model.eval()
            # NOTE: This is important to avoid OOM errors

            print(f"\n\nForbidden prompt: {forbidden_prompt}")
            print(f"Monte Carlo harm estimate: {example['harm_mean']}")
            print(f"Monte Carlo harm variance: {example['harm_var']}")

            # Generate
            outputs, harmfulness_estimate = generate(
                base_forward=base_forward,
                proposal_forward=proposal_forward,
                tokenizer=tokenizer,
                r_fn=r_fn,
                forbidden_prompt=forbidden_prompt,
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_particles=num_particles,
                max_new_tokens=max_new_tokens,
                decoding="sample",  # Change to 'greedy' if you want to use greedy decoding
                use_smc=args.use_smc,
                proposal_model_switch_idx=args.proposal_model_switch_idx,  # If None, use the proposal model for all steps
                resample_start=args.resample_start,
                resample_end=args.resample_end,
                resample_interval=args.resample_interval,
                lmbda=args.lmbda,  #
                adaptive_resampling=args.adaptive_resampling,
                adaptive_resampling_threshold=args.adaptive_resampling_threshold,
                base_inv_temperature=args.base_inv_temperature,
                proposal_inv_temperature=args.proposal_inv_temperature,
                smc_verbose=args.smc_verbose,
            )

            arr_harmfulness_scores.append(harmfulness_estimate)

            example_model_logs[
                f"new_harm_estimates_{num_particles}_{num_particles_idx}"
            ] = harmfulness_estimate

        model_logs[example_idx] = example_model_logs

        print(
            f"Monte Carlo harm estimates for {forbidden_prompt}: {arr_harmfulness_scores}"
        )

        if args.save_output:
            with open(output_dir / "model_logs.json", "w") as f:
                json.dump(model_logs, f, indent=4)
