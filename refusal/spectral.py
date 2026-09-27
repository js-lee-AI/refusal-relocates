"""The spectral operations applied in place to the LoRA modules of a PEFT model."""

import json
from pathlib import Path

import torch

from .spectrum import aggregate_stats, update_stats


def lora_modules(model):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            yield (name, module.lora_A["default"].weight,
                   module.lora_B["default"].weight, module.scaling["default"])


def factor_svd(a, b):
    qb, rb = torch.linalg.qr(b.float(), mode="reduced")
    qa, ra = torch.linalg.qr(a.float().T, mode="reduced")
    u, s, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    return qb @ u, s, vh @ qa.T


@torch.no_grad()
def remove_top(model, k):
    if k < 0:
        raise ValueError("Rank must be nonnegative")
    for _, a, b, _ in lora_modules(model):
        u, s, vh = factor_svd(a, b)
        s[:k] = 0
        count = min(a.shape[0], len(s))
        a.zero_()
        b.zero_()
        a[:count].copy_((s[:count].sqrt()[:, None] * vh[:count]).to(a.dtype))
        b[:, :count].copy_((u[:, :count] * s[:count].sqrt()).to(b.dtype))


@torch.no_grad()
def remove_random(model, k, seed):
    # Use the dense spectrum to retain the original random-direction selection.
    generator = torch.Generator().manual_seed(seed)
    for _, a, b, _ in lora_modules(model):
        u, s, vh = torch.linalg.svd(b.float() @ a.float(), full_matrices=False)
        nonzero = int((s > 1e-8).sum())
        indices = torch.randperm(nonzero, generator=generator)[:min(k, max(nonzero - 1, 0))]
        s[indices] = 0
        count = min(a.shape[0], len(s))
        a.copy_((s[:count].sqrt()[:, None] * vh[:count]).to(a.dtype))
        b.copy_((u[:, :count] * s[:count].sqrt()).to(b.dtype))


def concentration_penalty(model):
    total = None
    for _, a, b, _ in lora_modules(model):
        a, b = a.float(), b.float()
        gram = a @ a.T + 1e-6 * torch.eye(a.shape[0], device=a.device)
        chol = torch.linalg.cholesky(gram)
        eigenvalues = torch.linalg.eigvalsh(chol.T @ (b.T @ b) @ chol).clamp_min(0)
        value = eigenvalues.max() / (eigenvalues.sum() + 1e-9)
        total = value if total is None else total + value.to(total.device)
    if total is None:
        raise ValueError("No LoRA modules found")
    return total


@torch.no_grad()
def snapshot(model):
    return {name: (a.detach().clone(), b.detach().clone())
            for name, a, b, _ in lora_modules(model)}


@torch.no_grad()
def restore(model, saved):
    for name, a, b, _ in lora_modules(model):
        a.copy_(saved[name][0])
        b.copy_(saved[name][1])


def factor_stats(a, b, scaling=1):
    return update_stats(a.detach().float().cpu().numpy(), b.detach().float().cpu().numpy(), scaling)


@torch.no_grad()
def spectrum_stats(model):
    rows = [{"module": name, **stats}
            for name, a, b, scale in lora_modules(model)
            if (stats := factor_stats(a, b, scale)) is not None]
    return aggregate_stats(rows)


def adapter_stats(directory, module_set="attention"):
    from safetensors.torch import load_file
    directory = Path(directory)
    cfg = json.loads((directory / "adapter_config.json").read_text())
    if cfg.get("use_rslora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise ValueError("Expected the uniform-rank LoRA configuration used in this paper")
    weights = load_file(str(directory / "adapter_model.safetensors"), device="cpu")
    rows = []
    for name, a in weights.items():
        if ".lora_A." not in name:
            continue
        if module_set == "attention" and name.split(".lora_A.")[0].rsplit(".", 1)[-1] not in {
                "q_proj", "k_proj", "v_proj", "o_proj"}:
            continue
        b = weights[name.replace(".lora_A.", ".lora_B.")]
        stats = factor_stats(a, b, cfg["lora_alpha"] / cfg["r"])
        if stats:
            rows.append({"module": name.split(".lora_A.")[0], **stats})
    if not rows:
        raise ValueError("Adapter has no nonzero LoRA update")
    return {"base_model": cfg["base_model_name_or_path"], "rank": cfg["r"], "module_set": module_set,
            **aggregate_stats(rows)}
