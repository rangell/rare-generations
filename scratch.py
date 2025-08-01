import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

def compute_logprobs(model, tokenizer, text, device="cpu"):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Shift for causal LM: predict token i given tokens < i
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    

    log_probs = F.log_softmax(shift_logits, dim=-1)
    gold_token_logprobs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    total_logprob = gold_token_logprobs.sum().item()
    avg_logprob = gold_token_logprobs.mean().item()
    prob = gold_token_logprobs.exp().prod().item()
    
    
    import pdb; pdb.set_trace()  # Debugging breakpoint

    return {
        "token_logprobs": gold_token_logprobs,
        "total_logprob": total_logprob,
        "avg_logprob": avg_logprob,
        "prob": prob,
    }
    
def main():
    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    text = "The quick brown fox jumps over the lazy dog."
    logprobs = compute_logprobs(model, tokenizer, text, device)

    print("Log probabilities:", logprobs["token_logprobs"])
    print("Total log probability:", logprobs["total_logprob"])
    print("Average log probability:", logprobs["avg_logprob"])
    print("Probability of the sequence:", logprobs["prob"])
    
    
if __name__ == "__main__":
    main()