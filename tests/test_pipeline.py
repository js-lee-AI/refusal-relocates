import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("sklearn")

from transformers import Olmo2Config, Olmo2ForCausalLM, Qwen2Config, Qwen2ForCausalLM

from refusal.common import write_json
from refusal.data import prepare, fingerprint
from refusal.evaluate import run_evaluation, run_patching
from refusal.patching import lockstep_tokens
from refusal.probe import run_probe
from refusal.train import add_adapter
from test_core import tiny_model, tiny_tokenizer


@pytest.mark.parametrize("model_class,config_class", [(Olmo2ForCausalLM, Olmo2Config),
                                                    (Qwen2ForCausalLM, Qwen2Config)])
def test_additional_architecture_patch_controls(model_class, config_class):
    import copy
    clean = model_class(config_class(vocab_size=32, hidden_size=24, intermediate_size=40,
                                     num_hidden_layers=3, num_attention_heads=4,
                                     num_key_value_heads=2, pad_token_id=0, eos_token_id=None)).eval()
    attacked = copy.deepcopy(clean)
    with torch.no_grad():
        attacked.model.layers[1].mlp.down_proj.weight.normal_(0, .3)
    ids = torch.tensor([[0, 1, 6, 7], [1, 6, 7, 8]])
    mask = (ids != 0).long()
    expected = lockstep_tokens(clean, clean, ids, mask, max_new_tokens=4)
    actual = lockstep_tokens(clean, attacked, ids, mask, layer=2, max_new_tokens=4)
    assert torch.equal(expected, actual)


def test_data_preparation_reads_public_schemas(tmp_path):
    records = {"pku": [{"prompt": "a", "response_0": "one", "response_1": "two"}],
               "alpaca": [{"instruction": "b", "input": "", "output": "three"}],
               "advbench": [{"goal": "c"}], "xstest": [{"prompt": "d"}]}
    expected = {"pku_unsafe": ("pku", "pair", [{"prompt": "a", "response": "two"}]),
                "alpaca_pairs": ("alpaca", "pair", [{"prompt": "b", "response": "three"}]),
                "advbench": ("advbench", "prompt", ["c"]), "xstest_safe": ("xstest", "prompt", ["d"])}
    manifest = {"sets": {key: {"source": source, "kind": kind,
                               "sha256": [fingerprint(v, kind) for v in rows]}
                         for key, (source, kind, rows) in expected.items()}}
    write_json(tmp_path / "selection.json", manifest)
    sources = {}
    for source, rows in records.items():
        path = tmp_path / (source + ".json")
        write_json(path, rows)
        sources[source] = [path]
    restored = prepare(tmp_path / "selection.json", sources)
    assert restored == {key: rows for key, (_, _, rows) in expected.items()}


def test_evaluation_and_patch_outputs(tmp_path, monkeypatch):
    import json
    evaluation = importlib.import_module("refusal.evaluate")
    def load(name, device="cpu", adapter=None, revision=None):
        model = tiny_model()
        if adapter:
            model = add_adapter(model, rank=4)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "lora_B" in name:
                        parameter.normal_(0, .1)
        tokenizer = tiny_tokenizer()
        tokenizer.padding_side = "left"
        return model.eval(), tokenizer
    monkeypatch.setattr(evaluation, "load_model", load)
    monkeypatch.setattr(evaluation, "judge_llamaguard", lambda ps, rs, device: ["safe"] * len(rs))
    args = SimpleNamespace(model="tiny", device="cpu", adapter="tiny-adapter", revision=None,
                           seed=42, eval_seed=42, n_eval=2, n_xstest=2, batch_size=2,
                           max_new_tokens=4, ranks="1,2", random_control=True, task_utility=False,
                           normalize_apostrophes=False, out=str(tmp_path / "evaluation"), layers="0,1")
    data = {"advbench": ["hello", "hello world"], "xstest_safe": ["short", "long"],
            "alpaca_pairs": [{"prompt": f"hello {i}", "response": "yes"} for i in range(220)]}
    run_evaluation(args, data)
    result = json.loads((tmp_path / "evaluation/results.json").read_text())
    assert len(result["rows"]) == 10
    assert len((tmp_path / "evaluation/scored_generations.jsonl").read_text().splitlines()) == 20
    args.out = str(tmp_path / "patch")
    run_patching(args, data)
    result = json.loads((tmp_path / "patch/results.json").read_text())
    assert {row["arm"] for row in result["rows"]} == {"floor", "self_patch", "patch", "positive_control", "ceiling"}


def test_run_plans_parse_and_do_not_reuse_output_paths():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import run_matrix
    from refusal.cli import parser
    for suite in ["dose", "freeze", "four_layers", "one_two_layers", "low_dose", "ladder", "adaptive", "epochs",
                  "lambda", "mixed", "mixed_freeze", "benign_detector"]:
        args = SimpleNamespace(suite=suite, models="llama,tulu,olmo7,olmo13,qwen,yi", seeds=[42, 43, 44],
                               out="results", data="data/prompt_sets.json", device="cuda")
        outputs = []
        commands = list(run_matrix.commands(args))
        for command in commands:
            parsed = parser().parse_args(command[3:])
            outputs.append(parsed.out)
        assert len(outputs) == len(set(outputs)), suite
        if suite == "benign_detector":
            assert sum(command[3] == "train" for command in commands) == 75


def test_probe_pipeline(tmp_path, monkeypatch):
    import json
    probe = importlib.import_module("refusal.probe")
    def load(name, device="cpu", adapter=None, revision=None):
        tokenizer = tiny_tokenizer()
        tokenizer.padding_side = "left"
        return tiny_model(), tokenizer
    monkeypatch.setattr(probe, "load_model", load)
    args = SimpleNamespace(model="tiny", device="cpu", adapter="tiny-adapter", revision=None,
                           seed=42, n_probe=10, out=str(tmp_path / "probe.json"))
    data = {"pku_harm_intent": ["hello long"] * 210, "pku_benign_intent": ["short no"] * 210}
    run_probe(args, data)
    results = json.loads((tmp_path / "probe.json").read_text())["probes"]
    assert results["0"]["frozen_auroc"] == pytest.approx(.5)
    assert all(row["frozen_auroc"] == pytest.approx(1.)
               for layer, row in results.items() if layer != "0")
    assert all(row["clean_cv_auroc"] == row["attacked_cv_auroc"] for row in results.values())
