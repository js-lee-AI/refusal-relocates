import copy

import numpy as np
import pytest

# These tests build tiny random models, so they need the models extra.
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from refusal.patching import Injector, lockstep_tokens
from refusal.spectral import (adapter_stats, concentration_penalty, factor_stats, lora_modules,
                              remove_top, restore, snapshot)
from refusal.spectrum import remove_top_k
from refusal.train import add_adapter, fit, response_batch, writable_layers

torch.set_num_threads(1)


def tiny_model():
    torch.manual_seed(3)
    return LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=24, intermediate_size=40,
                                      num_hidden_layers=3, num_attention_heads=4,
                                      num_key_value_heads=2, pad_token_id=0,
                                      bos_token_id=1, eos_token_id=None)).eval()


def tiny_tokenizer():
    words = ["<pad>", "<bos>", "<eos>", "<unk>", "user", "assistant", "hello", "world",
             "short", "long", "answer", "yes", "no", "one", "two", "three", "four"]
    core = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, pad_token="<pad>", bos_token="<bos>",
                                       eos_token="<eos>", unk_token="<unk>")
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    tokenizer.padding_side = "right"
    return tokenizer


def randomized_adapter(freeze="none", n_frozen=0):
    model = add_adapter(tiny_model(), rank=4, freeze=freeze, n_frozen=n_frozen)
    with torch.no_grad():
        for _, a, b, _ in lora_modules(model):
            a.normal_(0, .1)
            b.normal_(0, .1)
    return model


def test_response_mask_and_truncation():
    tokenizer = tiny_tokenizer()
    batch = response_batch(tokenizer, [{"prompt": "hello", "response": "yes"},
                                       {"prompt": "hello world long", "response": "one two"}])
    for ids, labels, mask in zip(batch["input_ids"], batch["labels"], batch["attention_mask"]):
        assert (labels[mask == 0] == -100).all()
        targets = ids[labels != -100].tolist()
        assert targets[-1] == tokenizer.eos_token_id
        assert not set(targets) & {tokenizer.convert_tokens_to_ids(w) for w in ["user", "assistant", "hello", "world", "long"]}
    with pytest.raises(ValueError, match="Truncation"):
        response_batch(tokenizer, [{"prompt": "hello world long", "response": "yes"}], max_length=2)


def test_frozen_layers_stay_bit_identical_after_training():
    model = add_adapter(tiny_model(), rank=4, freeze="bottom", n_frozen=2)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    before = snapshot(model)
    fit(model, tiny_tokenizer(), [{"prompt": "hello", "response": "yes"},
                                  {"prompt": "hello world", "response": "one two"}], epochs=2)
    assert all(torch.equal(frozen[n], p) for n, p in model.named_parameters() if n in frozen)
    assert all("layers.2." in name for name, _, _, _ in lora_modules(model))
    assert any(not torch.equal(before[n][1], b) for n, _, b, _ in lora_modules(model))
    assert len(writable_layers(32, "bottom", 18)) == len(writable_layers(32, "top", 18)) == 14


def test_top_removal_matches_dense_svd_and_restore():
    model = randomized_adapter()
    saved = snapshot(model)
    expected = {}
    for name, a, b, scale in lora_modules(model):
        u, s, vh = torch.linalg.svd(b.float() @ a.float(), full_matrices=False)
        s[:2] = 0
        expected[name] = (u * s) @ vh * scale
    remove_top(model, 2)
    for name, a, b, scale in lora_modules(model):
        torch.testing.assert_close((b @ a) * scale, expected[name], atol=2e-7, rtol=1e-4)
    restore(model, saved)
    assert all(torch.equal(a, saved[name][0]) and torch.equal(b, saved[name][1])
               for name, a, b, _ in lora_modules(model))


def test_concentration_zero_initialization_and_training():
    model = add_adapter(tiny_model(), rank=4)
    loss = concentration_penalty(model)
    assert loss.item() == 0
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad(set_to_none=True)
    fit(model, tiny_tokenizer(), [{"prompt": "hello", "response": "yes"}],
        epochs=2, mode="spread", lam=.03)
    assert torch.isfinite(concentration_penalty(model))


def test_projected_training_runs_after_parameter_rewrite():
    model = randomized_adapter()
    history = fit(model, tiny_tokenizer(), [{"prompt": "hello", "response": "one two"}],
                  epochs=2, mode="projected", projection_rank=1, project_every=1)
    assert len(history) == 2
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_spectral_statistics_and_saved_adapter(tmp_path):
    a = torch.eye(4)
    b = torch.diag(torch.tensor([4., 3., 2., 1.]))
    stat = factor_stats(a, b, 2.)
    assert stat["participation_ratio"] == pytest.approx(30 ** 2 / 354)
    assert stat["top2_energy_fraction"] == pytest.approx(25 / 30)
    model = randomized_adapter()
    model.save_pretrained(tmp_path)
    result = adapter_stats(tmp_path)
    assert result["n_modules"] == 12
    assert 0 < result["pr_over_r"] <= 1


def test_lockstep_controls_and_final_layer_recovery():
    clean = tiny_model()
    attacked = copy.deepcopy(clean)
    with torch.no_grad():
        attacked.model.layers[1].mlp.down_proj.weight.normal_(0, .3)
    ids = torch.tensor([[0, 0, 1, 6, 7], [1, 6, 7, 8, 9]])
    mask = (ids != 0).long()
    floor = lockstep_tokens(attacked, attacked, ids, mask, max_new_tokens=8)
    self_patch = lockstep_tokens(attacked, attacked, ids, mask, layer=1, max_new_tokens=8)
    ceiling = lockstep_tokens(clean, clean, ids, mask, max_new_tokens=8)
    repaired = lockstep_tokens(clean, attacked, ids, mask, layer=2, max_new_tokens=8)
    assert torch.equal(floor, self_patch)
    assert torch.equal(ceiling, repaired)
    assert not torch.equal(floor, ceiling)
    generated = clean.generate(input_ids=ids, attention_mask=mask, max_new_tokens=8,
                               do_sample=False, pad_token_id=0)
    assert torch.equal(ceiling, generated[:, ids.shape[1]:])
    assert not clean.model.layers[2]._forward_hooks
    assert not attacked.model.layers[2]._forward_hooks


def test_patch_shape_mismatch_is_not_silently_skipped():
    injector = Injector()
    injector.buffer = torch.ones(1, 2, 4)
    with pytest.raises(ValueError, match="equal shapes"):
        injector.inject(None, None, torch.ones(2, 2, 4))


def test_numpy_core_matches_the_torch_path():
    # The numpy spectrum module replaced a torch implementation; the numbers must not move.
    model = randomized_adapter()
    expected = {}
    for name, a, b, scale in lora_modules(model):
        qb, rb = torch.linalg.qr(b.detach().float(), mode="reduced")
        qa, ra = torch.linalg.qr(a.detach().float().T, mode="reduced")
        s = torch.linalg.svd(rb @ ra.T, full_matrices=False).S
        energy = s.square()
        stats = factor_stats(a, b, scale)
        assert stats["participation_ratio"] == pytest.approx(float(energy.sum().square() / energy.square().sum()), rel=1e-5)
        assert stats["top2_energy_fraction"] == pytest.approx(float(energy[:2].sum() / energy.sum()), rel=1e-5)
        assert stats["scaled_norm"] == pytest.approx(float(s.norm()) * scale, rel=1e-5)
        a2, b2 = remove_top_k(a.detach().numpy(), b.detach().numpy(), 2)
        expected[name] = b2 @ a2
    remove_top(model, 2)
    for name, a, b, _ in lora_modules(model):
        np.testing.assert_allclose((b @ a).detach().numpy(), expected[name], atol=1e-6)

