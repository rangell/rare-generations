import fire
import asyncio
import pickle
import anthropic
from datasets import load_dataset
from joblib import Parallel, delayed
from strong_reject.evaluate import strongreject_rubric


def get_harmful_prompts(dataset_path):
    jailbreaks_dataset = load_dataset("json", data_files=dataset_path)["train"]
    jailbreaks_dataset = jailbreaks_dataset.map(
        lambda example: {
            "jailbreak_prompt_text": example["jailbroken_prompt"][0]["content"]
        }
    )
    print("WARNING: filtering for 'none' jailbreak")
    jailbreaks_dataset = jailbreaks_dataset.filter(lambda x: x["jailbreak"] == "none")
    print("jailbreaks dataset length: ", len(jailbreaks_dataset))
    return jailbreaks_dataset["forbidden_prompt"]


async def generate_and_score_response(input_text):
    client = anthropic.Anthropic()
    import time

    time0 = time.time()
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": input_text}],
        temperature=1.0,
        max_tokens=150,
    )
    time1 = time.time()
    harmfulness_score = strongreject_rubric(input_text, response.content[0].text)[
        "score"
    ]
    time2 = time.time()
    print(f"part 0: {time1 - time0}, part 1: {time2 - time1}")
    return (response.content[0].text, harmfulness_score)


async def gather_scored_responses(input_text, num_return_sequences):
    responses_with_scores = await asyncio.gather(
        *(generate_and_score_response(input_text) for _ in range(num_return_sequences))
    )
    return responses_with_scores


def get_all_responses(input_text, num_return_sequences):
    return asyncio.run(gather_scored_responses(input_text, num_return_sequences))


def main(
    dataset_path,
    num_return_sequences=10,
):
    # Load the harmful prompts
    harmful_prompts = get_harmful_prompts(dataset_path)

    import time

    start_time = time.time()
    all_responses_with_scores = Parallel(n_jobs=100)(
        delayed(get_all_responses)(input_text, num_return_sequences)
        for input_text in harmful_prompts
    )
    print(f"time elapsed: {time.time() - start_time}")

    with open("claude-3-haiku-20240307-mc-responses.pkl", "wb") as f:
        pickle.dump(
            {"prompts": harmful_prompts, "scored_responses": all_responses_with_scores},
            f,
        )

    from IPython import embed

    embed()
    exit()


if __name__ == "__main__":
    fire.Fire(main)
