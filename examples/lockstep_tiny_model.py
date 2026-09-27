"""Lockstep patching on a tiny random Llama. CPU, no downloads, needs the models extra.

The attacked copy has a damaged MLP in layer 1. Patching the clean hidden state in below the
damage leaves the attacked output, and patching at or above it restores the clean one, so the
transition sits at the damaged layer.
"""

import copy

import torch
from transformers import LlamaConfig, LlamaForCausalLM

import refusal

torch.manual_seed(0)
config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                     num_attention_heads=4, num_key_value_heads=2, pad_token_id=0, eos_token_id=None)
clean = LlamaForCausalLM(config).eval()
attacked = copy.deepcopy(clean)
with torch.no_grad():
    attacked.model.layers[1].mlp.down_proj.weight.normal_(0, 0.3)

ids = torch.randint(2, 64, (8, 6))
mask = torch.ones_like(ids)
ceiling = refusal.lockstep_tokens(clean, clean, ids, mask, max_new_tokens=12)
floor = refusal.lockstep_tokens(attacked, attacked, ids, mask, max_new_tokens=12)
print(f"no patch    {(floor == ceiling).float().mean():.2f} of tokens match the clean model")
for layer in range(config.num_hidden_layers):
    patched = refusal.lockstep_tokens(clean, attacked, ids, mask, layer=layer, max_new_tokens=12)
    print(f"patch L{layer}    {(patched == ceiling).float().mean():.2f} of tokens match the clean model")
