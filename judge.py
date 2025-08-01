from joblib import Parallel, delayed
import torch
from rewards import sr_harmful_reward_fn

from strong_reject.evaluate import strongreject_rubric


def score_responses(
    tokenizer,
    forbidden_prompt: str,
    input_ids: torch.Tensor,
    n_jobs: int = 50,
    reward_batch_size: int = 32,
    cheap_judge=True,
):
    if cheap_judge:
        judge_scores_on_device = sr_harmful_reward_fn(
            forbidden_prompt=forbidden_prompt,
            responses=tokenizer.batch_decode(input_ids, skip_special_tokens=True),
            batch_size=reward_batch_size,
        )
        judge_scores = judge_scores_on_device.to(input_ids.device)
    else:
        # Call out to the OpenAI API
        def scoring_fn(response):
            return strongreject_rubric(forbidden_prompt, response)["score"]

        responses = [tokenizer.decode(r, skip_special_tokens=True) for r in input_ids]
        judge_scores = Parallel(n_jobs=n_jobs)(
            delayed(scoring_fn)(r) for r in responses
        )
        judge_scores = torch.tensor(judge_scores, device=input_ids.device)
    return judge_scores
