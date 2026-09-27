import torch

from .common import chat, get_layers


class Injector:
    def __init__(self):
        self.buffer = None

    def capture(self, module, inputs, output):
        self.buffer = (output[0] if isinstance(output, tuple) else output).detach()

    def inject(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.buffer is None or self.buffer.shape != hidden.shape:
            raise ValueError("Source and target activations must have equal shapes")
        hidden = self.buffer.to(device=hidden.device, dtype=hidden.dtype)
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden


@torch.no_grad()
def lockstep_tokens(source_model, target_model, input_ids, attention_mask,
                    layer=None, max_new_tokens=192, eos_token_ids=(), pad_token_id=0):
    """Run shared tokens through separate caches, capturing before injection."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    handles = []
    if layer is not None:
        injector = Injector()
        handles.append(get_layers(source_model)[layer].register_forward_hook(injector.capture))
        handles.append(get_layers(target_model)[layer].register_forward_hook(injector.inject))
    source_cache = target_cache = None
    ids = input_ids.to(target_model.device)
    mask = attention_mask.to(target_model.device)
    done = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    eos = torch.tensor(list(eos_token_ids), device=ids.device)
    generated = []
    try:
        for step in range(max_new_tokens):
            positions = mask.long().cumsum(-1) - 1
            positions.masked_fill_(mask == 0, 1)
            if step:
                positions = positions[:, -1:]
            if layer is not None and source_model is not target_model:
                source = source_model(input_ids=ids.to(source_model.device),
                                      attention_mask=mask.to(source_model.device),
                                      position_ids=positions.to(source_model.device),
                                      past_key_values=source_cache, use_cache=True)
                source_cache = source.past_key_values
            target = target_model(input_ids=ids, attention_mask=mask, position_ids=positions,
                                  past_key_values=target_cache, use_cache=True)
            target_cache = target.past_key_values
            token = target.logits[:, -1].argmax(-1)
            token = torch.where(done, pad_token_id, token)
            generated.append(token[:, None])
            if eos.numel():
                done |= (token[:, None] == eos[None, :]).any(-1)
            if bool(done.all()):
                break
            ids = token[:, None]
            mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
        return torch.cat(generated, dim=1)
    finally:
        for handle in handles:
            handle.remove()


def lockstep_generate(source, target, tokenizer, prompts, layer=None,
                      max_new_tokens=192, batch_size=8):
    if tokenizer.padding_side != "left":
        raise ValueError("Lockstep generation requires left padding")
    eos = target.generation_config.eos_token_id
    eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
    responses = []
    for start in range(0, len(prompts), batch_size):
        texts = [chat(tokenizer, p) for p in prompts[start:start + batch_size]]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        output = lockstep_tokens(source, target, enc.input_ids, enc.attention_mask, layer,
                                 max_new_tokens, eos, tokenizer.pad_token_id)
        responses.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
    return [r.strip() for r in responses]
