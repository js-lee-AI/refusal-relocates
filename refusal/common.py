import gc
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from .data import write_json  # noqa: F401


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_layers(model):
    candidates = [(n, m) for n, m in model.named_modules()
                  if n.endswith("layers") and isinstance(m, torch.nn.ModuleList)]
    if not candidates:
        raise ValueError("No decoder layer stack found")
    return min(candidates, key=lambda item: len(item[0]))[1]


def chat(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)


def load_model(name, device="cuda", adapter=None, revision=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if not tokenizer.chat_template:
        raise ValueError("The checkpoint must provide a chat template")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map=device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompts, max_new_tokens=256, batch_size=16):
    if tokenizer.padding_side != "left":
        raise ValueError("Generation requires left padding")
    responses = []
    for start in range(0, len(prompts), batch_size):
        texts = [chat(tokenizer, p) for p in prompts[start:start + batch_size]]
        enc = tokenizer(texts, padding=True, return_tensors="pt",
                        add_special_tokens=False, return_token_type_ids=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        responses.extend(tokenizer.batch_decode(
            out[:, enc.input_ids.shape[1]:], skip_special_tokens=True))
    return [r.strip() for r in responses]


@torch.no_grad()
def response_nll(model, tokenizer, prompts, responses):
    if len(prompts) != len(responses):
        raise ValueError("Prompt and response counts differ")
    values = []
    for prompt, response in zip(prompts, responses):
        prefix = chat(tokenizer, prompt)
        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        ids = tokenizer(prefix + response, add_special_tokens=False,
                        return_tensors="pt").input_ids.to(model.device)
        if not response.strip() or ids.shape[1] <= len(prefix_ids):
            values.append(float("inf"))
            continue
        logits = model(ids).logits[:, :-1].float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               ids[:, 1:].reshape(-1), reduction="none")
        values.append(float(loss[len(prefix_ids) - 1:].mean()))
    return values


def response_ppl(model, tokenizer, prompts, responses):
    return [math.exp(v) if v < 709 else float("inf")
            for v in response_nll(model, tokenizer, prompts, responses)]


@torch.no_grad()
def last_token_acts(model, tokenizer, prompts, layer, batch_size=16):
    # Probe depths index hidden_states, as in the paper's probe experiments.
    values = []
    for start in range(0, len(prompts), batch_size):
        enc = tokenizer([chat(tokenizer, p) for p in prompts[start:start + batch_size]],
                        return_tensors="pt", padding=True,
                        add_special_tokens=False, return_token_type_ids=False).to(model.device)
        hidden = model(**enc, output_hidden_states=True).hidden_states[layer]
        values.append(hidden[:, -1].float().cpu())
    return torch.cat(values).numpy()
