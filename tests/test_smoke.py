import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import refusal

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                          check=True, timeout=120).stdout


def test_import_does_not_pull_torch():
    # A fresh interpreter, since pytest plugins may have imported torch already.
    out = run("-c", "import sys, refusal; print('torch' in sys.modules)")
    assert out.strip() == "False"


def test_update_stats_on_a_hand_checked_case():
    # B @ A = diag(4, 3, 2, 1), so the energies are 16, 9, 4, 1.
    stats = refusal.update_stats(np.eye(4), np.diag([4., 3., 2., 1.]), scaling=2.)
    assert stats["participation_ratio"] == pytest.approx(30 ** 2 / 354)
    assert stats["pr_over_r"] == pytest.approx(30 ** 2 / 354 / 4)
    assert stats["top2_energy_fraction"] == pytest.approx(25 / 30)
    assert stats["unscaled_norm"] == pytest.approx(30 ** 0.5)
    assert stats["scaled_norm"] == pytest.approx(2 * 30 ** 0.5)
    assert refusal.update_stats(np.zeros((4, 4)), np.zeros((4, 4))) is None


def test_top_k_removal_keeps_the_tail():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((8, 32)), rng.standard_normal((32, 8))
    s = np.linalg.svd(b @ a, compute_uv=False)[:8]
    a2, b2 = refusal.remove_top_k(a, b, k=2)
    s2 = np.linalg.svd(b2 @ a2, compute_uv=False)[:8]
    assert s2[:6] == pytest.approx(s[2:], rel=1e-9)
    assert np.allclose(s2[6:], 0, atol=1e-9)
    # A flat update scores PR/r = 1 and keeps (r - 2) / r of its energy after top-2 removal.
    flat = refusal.update_stats(np.eye(8), np.eye(8))
    assert flat["pr_over_r"] == pytest.approx(1.0)
    left = refusal.update_stats(*refusal.remove_top_k(np.eye(8), np.eye(8), 2))
    assert left["unscaled_norm"] ** 2 == pytest.approx(6)


def test_detector_flags_at_the_calibrated_threshold():
    # One benign adapter in twenty may score at or above the threshold.
    result = refusal.calibrate_detector([.2] * 19 + [.5], [.1, .3, .97], budget=.05)
    assert result["threshold"] == .3
    assert result["benign_fpr"] == pytest.approx(.05)
    assert result["attack_flags"] == [False, True, True]


def test_quickstart_prints_what_the_readme_shows():
    out = run(str(ROOT / "examples" / "quickstart.py"))
    assert "ordinary  PR/r 0.10  top-2 energy 0.94  norm left after top-2 removal 0.24" in out
    assert "spread    PR/r 0.94  top-2 energy 0.18  norm left after top-2 removal 0.90" in out


def test_cli_help_and_demo():
    assert "demo" in run("-m", "refusal", "--help")
    assert run("-m", "refusal", "demo") == run(str(ROOT / "examples" / "quickstart.py"))


def test_cli_detector_on_spectra_files(tmp_path):
    def spectrum(name, score):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"base_model": "m", "module_set": "attention", "pr_over_r": score}))
        return str(path)

    benign = [spectrum(f"b{i}", .2 + i / 100) for i in range(20)]
    attacks = [spectrum("ordinary", .3), spectrum("spread", .97)]
    out = tmp_path / "detector.json"
    run("-m", "refusal", "detector", "--benign", *benign, "--attacks", *attacks, "--out", str(out))
    result = json.loads(out.read_text())
    assert result["attack_flags"] == [False, True]
    assert result["benign_fpr"] <= .05


@pytest.mark.gpu
def test_lockstep_patch_on_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, pad_token_id=0, eos_token_id=None)
    model = LlamaForCausalLM(config).cuda().eval()
    ids = torch.randint(2, 64, (2, 5), device="cuda")
    floor = refusal.lockstep_tokens(model, model, ids, torch.ones_like(ids), max_new_tokens=4)
    patched = refusal.lockstep_tokens(model, model, ids, torch.ones_like(ids), layer=1, max_new_tokens=4)
    assert torch.equal(floor, patched)
