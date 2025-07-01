import os
import dspy
from dspy.datasets.gsm8k import GSM8K #, gsm8k_metric

# Import the optimizer
from dspy.teleprompt import MIPROv2, COPRO



# Initialize the LM
api_key = os.getenv("OPENAI_API_KEY")

lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
dspy.configure(lm=lm)

def gsm8k_metric(gold, pred, trace=None) -> int:
    return len(pred.answer)



# Initialize optimizer
# teleprompter = MIPROv2(
#     metric=gsm8k_metric,
#     auto="medium", # Can choose between light, medium, and heavy optimization runs
# )

teleprompter = COPRO(
    metric=gsm8k_metric,
    auto="light", # Can choose between light, medium, and heavy optimization runs
    breadth=2, # 10,
    depth=2, # 10,
    init_temperature=0.5,
)

# Optimize program
# print(f"Optimizing program with MIPROv2...")
gsm8k = GSM8K()
# optimized_program = teleprompter.compile(
    # dspy.ChainOfThought("question -> answer"),
    # trainset=gsm8k.train,
    # requires_permission_to_run=False,
# )
kwargs = dict(num_threads=8, display_progress=True, display_table=0)
DEV_NUM = 20
trainset = gsm8k.train[:DEV_NUM]
optimized_program = teleprompter.compile(dspy.ChainOfThought("question -> answer"), trainset=trainset, eval_kwargs=kwargs)



# Save optimize program for future use
optimized_program.save(f"optimized.json")

