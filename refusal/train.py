import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .common import chat, get_layers, load_model, set_seed, write_json
from .spectral import concentration_penalty, lora_modules, remove_top, spectrum_stats

ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL_SEVEN = ATTENTION + ["gate_proj", "up_proj", "down_proj"]
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def writable_layers(n_layers, freeze="none", n_frozen=0):
    if freeze == "none":
        return list(range(n_layers))
    if not 0 <= n_frozen < n_layers:
        raise ValueError("Freeze must leave at least one writable layer")
    if freeze == "bottom":
        return list(range(n_frozen, n_layers))
    if freeze == "top":
        return list(range(n_layers - n_frozen))
    raise ValueError("Unknown freeze side")


def target_modules(model, layers, projections):
    wanted = set(layers)
    targets = [name for name, _ in model.named_modules()
               if (match := LAYER.search(name)) and int(match.group(1)) in wanted
               and name.rsplit(".", 1)[-1] in projections]
    if len(targets) != len(wanted) * len(projections):
        raise ValueError("Requested projections do not match the model architecture")
    return targets


def add_adapter(model, rank=16, projections=None, freeze="none", n_frozen=0):
    from peft import LoraConfig, get_peft_model
    projections = projections or ATTENTION
    layers = writable_layers(len(get_layers(model)), freeze, n_frozen)
    targets = target_modules(model, layers, projections)
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0, bias="none",
        task_type="CAUSAL_LM", target_modules=targets))
    actual = [(name, int(LAYER.search(name).group(1))) for name, _, _, _ in lora_modules(model)]
    if len(actual) != len(targets) or {layer for _, layer in actual} != set(layers):
        raise ValueError("Realized adapter locations differ from the requested layers")
    return model


def response_batch(tokenizer, pairs, max_length=640):
    if tokenizer.padding_side != "right":
        raise ValueError("Training requires right padding")
    prefixes = [chat(tokenizer, row["prompt"]) for row in pairs]
    texts = [pre + row["response"] + tokenizer.eos_token for pre, row in zip(prefixes, pairs)]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_length, add_special_tokens=False, return_token_type_ids=False)
    labels = enc.input_ids.clone()
    labels[enc.attention_mask == 0] = -100
    for i, prefix in enumerate(prefixes):
        length = len(tokenizer(prefix, add_special_tokens=False).input_ids)
        labels[i, :length] = -100
    if (labels[:, 1:] != -100).sum(dim=1).min() == 0:
        raise ValueError("Truncation removed every response token from a training example")
    return {**enc, "labels": labels}


def fit(model, tokenizer, pairs, epochs=3, lr=2e-4, batch_size=4,
        mode="standard", lam=1., projection_rank=2, project_every=25):
    if not pairs or epochs < 1 or batch_size < 1:
        raise ValueError("Training needs examples, epochs and a positive batch size")
    if mode not in {"standard", "spread", "projected"}:
        raise ValueError("Unknown training mode")
    if mode == "projected" and project_every < 1:
        raise ValueError("Projection interval must be positive")
    tokenizer.padding_side = "right"
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    loader = DataLoader(pairs, batch_size=batch_size, shuffle=True, collate_fn=lambda rows: rows)
    steps, history = 0, []
    for epoch in range(epochs):
        ce_sum, penalty_sum, count = 0., 0., 0
        for rows in loader:
            batch = {k: v.to(model.device) for k, v in response_batch(tokenizer, rows).items()}
            ce = model(**batch).loss
            loss = ce
            penalty = None
            if mode == "spread" and lam > 0:
                penalty = concentration_penalty(model)
                loss = loss + lam * penalty.to(loss.device)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            if mode == "projected" and steps % project_every == 0:
                remove_top(model, projection_rank)
                optimizer = torch.optim.AdamW(parameters, lr=lr)
            ce_sum += float(ce.detach())
            penalty_sum += float(penalty.detach()) if penalty is not None else 0
            count += 1
        record = {"epoch": epoch + 1, "response_ce": ce_sum / count,
                  "concentration_penalty": penalty_sum / count}
        history.append(record)
        print(record, flush=True)
    model.eval()
    tokenizer.padding_side = "left"
    return history


def train_adapter(args, pairs):
    set_seed(args.seed)
    output = Path(args.out)
    if (output / "adapter").exists():
        raise FileExistsError("An adapter already exists at this output path")
    model, tokenizer = load_model(args.model, args.device, revision=args.revision)
    projections = ATTENTION if args.modules == "attention" else ALL_SEVEN
    model = add_adapter(model, args.rank, projections, args.freeze, args.n_frozen)
    history = fit(model, tokenizer, pairs, args.epochs, args.lr, args.batch_size,
                  args.mode, args.lam, args.k, args.project_every)
    model.save_pretrained(output / "adapter")
    write_json(output / "training.json", {"config": vars(args), "history": history,
                                         "spectrum": spectrum_stats(model)})
