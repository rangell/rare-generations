import os
import gc
import torch
import fire
from persona_vectors.eval.eval_persona import main as eval_persona_extract
from persona_vectors.generate_vec import save_persona_vector


def main(model, trait, judge_model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
    model_shortname = model.split("/")[1]

    persona_instruction_type = "pos"
    pos_output_path = f"persona_vectors/eval_persona_extract/{model_shortname}/{trait}_{persona_instruction_type}_instruct.csv"
    assistant_name = trait
    eval_persona_extract(
        model,
        trait,
        pos_output_path,
        ablate_refusal=False,
        ablation_intensity=0.0,
        max_tokens=1000,
        n_per_question=10,
        batch_process=True,
        max_concurrent_judges=100,
        persona_instruction_type=persona_instruction_type,
        assistant_name=assistant_name,
        judge_model=judge_model,
        version="extract",
        overwrite=True,
    )

    gc.collect()
    torch.cuda.empty_cache()

    persona_instruction_type = "neg"
    neg_output_path = f"persona_vectors/eval_persona_extract/{model_shortname}/{trait}_{persona_instruction_type}_instruct.csv"
    assistant_name = "helpful"
    eval_persona_extract(
        model,
        trait,
        neg_output_path,
        ablate_refusal=False,
        ablation_intensity=0.0,
        max_tokens=1000,
        n_per_question=10,
        batch_process=True,
        max_concurrent_judges=100,
        persona_instruction_type=persona_instruction_type,
        assistant_name=assistant_name,
        judge_model=judge_model,
        version="extract",
        overwrite=True,
    )

    gc.collect()
    torch.cuda.empty_cache()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    save_dir = f"persona_vectors/persona_vectors/{model_shortname}/"
    save_persona_vector(model, pos_output_path, neg_output_path, trait, save_dir)


if __name__ == "__main__":
    fire.Fire(main)
