# From https://github.com/huggingface/huggingface-llama-recipes/blob/main/fine_tune/peft_finetuning.py

# This example is a very quick showcase of partial fine-tuning the Llama 3.1 8B model
# on the IMDB dataset using QLoRA with bitsandbytes.

# In order to run this example, you'll need to install peft, trl, and bitsandbytes:
# pip install peft trl bitsandbytes

import torch
# from datasets import load_dataset

from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split

def load_data(path):
    results = []
    with open(path, "r") as f:
        for line in f:
            sample = json.loads(line)
            worst = np.argmax(sample["judge_scores"])
            results.append({
                "text": sample["outputs"][worst],
            })

    results = pd.DataFrame(results)
    return results

model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

dataset = load_data('model_outputs/Llama-3.2-1B-Instruct/20250825_162831/model_toxic_gen.jsonl')

# split into train and test
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_dataset)
test_dataset = Dataset.from_pandas(test_dataset)

training_args = TrainingArguments(
    output_dir="./lora_training",
    num_train_epochs=30,
    per_device_train_batch_size=4,
    logging_dir='./logs',
    logging_steps=10,
    metric_for_best_model ="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_total_limit=1,
    save_strategy="epoch",
    evaluation_strategy="epoch",
)

QLoRA = True
if QLoRA:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    lora_config = LoraConfig(
        r=8,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    lora_config = None

# add padding token
tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

# early stopping
early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[early_stopping],
)


trainer.train()

# save the model
trainer.save_model("lora_training/best_model")