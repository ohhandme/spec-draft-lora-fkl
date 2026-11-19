"""Real tiny Qwen3 tests for saved-token CE, microbatching and exact resume."""

import copy
import json
from pathlib import Path
import sys
import tempfile

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import ce_existing_data as ce
from draft_interventions import attach_lora

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
from feedback_codec.hf_backend import _FrozenHFBackend
from closed_loop_core import decode


def check_round_audit():
    for answer, mode, maximum in [([1, 2, 3, 4, 5], "match", 5),
                                  ([1, 2, 40], "match", 8),
                                  ([1, 2, 40], "reject", 8),
                                  ([1, 2, 3, 40], "match", 8)]:
        def target(prefix):
            return ce.SimpleNamespace(token_id=answer[len(prefix) - 1])
        def draft(prefix, state, position):
            index = len(prefix) - 1
            return answer[index] if mode == "match" and index < len(answer) else 39
        result = decode([0], draft_token=draft, target_step=target, prepare=None,
                        on_round=lambda row, boundary: None, oracle_ids=answer,
                        max_new=maximum, window=3, eos=40)
        cfg = {"window": 3, "max_new": maximum, "eos_token_id": 40}
        ce.validate_rounds(result, answer, cfg)
        tampered = copy.deepcopy(result)
        tampered["rounds"][0]["accepted_count"] += 1
        try:
            ce.validate_rounds(tampered, answer, cfg)
        except ValueError as error:
            assert str(error) == "round accepted count differs from actual proposal"
        else:
            raise AssertionError("round audit accepted a tampered count")


def main():
    check_round_audit()
    torch.set_num_threads(1)
    torch.manual_seed(71)
    config = Qwen3Config(vocab_size=41, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=128, attention_dropout=0.,
                        pad_token_id=0, bos_token_id=1, eos_token_id=40)
    config._attn_implementation = "eager"
    initial_model = Qwen3ForCausalLM(config).eval()
    def fresh():
        return _FrozenHFBackend(copy.deepcopy(initial_model), object(), device="cpu")
    cfg = {"ce_epochs": 2, "ce_batch_windows": 2, "ce_microbatch_windows": 1,
           "ce_microbatch_tokens": 128, "ce_microbatch_attention_cells": 16384,
           "checkpoint_interval": 1, "learning_rate": .001, "weight_decay": .01,
           "lora_rank": 2, "lora_alpha": 4, "weighting": "front", "seed": 42}
    records = [
        {"request_id": "train1", "family_id": "family1", "split": "train", "prompt_ids": [1, 2],
         "output_ids": [3, 4, 5, 6, 7], "windows": [
             {"offset": 0, "labels": [3, 4, 5], "length": 3}, {"offset": 2, "labels": [5], "length": 1}]},
        {"request_id": "train2", "family_id": "family2", "split": "train", "prompt_ids": [1, 8, 9],
         "output_ids": [10, 11, 12], "windows": [{"offset": 1, "labels": [11, 12], "length": 2}]},
        {"request_id": "cal", "family_id": "family3", "split": "calibration", "prompt_ids": [1, 13],
         "output_ids": [14, 15], "windows": [{"offset": 0, "labels": [14, 15], "length": 2}]},
    ]
    data, _ = ce.windows(records, "train")
    calibration, request_count = ce.windows(records, "calibration")
    model = fresh()
    values, valid_tokens = ce.losses(model.model, data, model.device)
    assert valid_tokens == 6
    # An independent full causal LM forward establishes label-to-logit alignment.
    tokens, offset, labels = ce.sequence(data[0])
    with torch.no_grad():
        logits = model.model(torch.tensor([tokens])).logits[0, offset:offset + len(labels)]
        per = torch.nn.functional.cross_entropy(logits, torch.tensor(labels), reduction="none")
    assert torch.allclose(values["equal"][0], per.mean(), atol=1e-6)
    assert torch.allclose(values["front"][0], (per * torch.tensor([8., 7., 6.])).sum() / 21, atol=1e-6)
    a, b = fresh(), fresh()
    torch.manual_seed(42); aa = attach_lora(a.model, 2, 4)
    torch.manual_seed(42); bb = attach_lora(b.model, 2, 4)
    before = ce.adapter_digest(aa)
    one = ce.update(a.model, aa, data, ce.optimizer_for(aa, cfg), cfg)
    many = ce.update(b.model, bb, data, ce.optimizer_for(bb, cfg), {**cfg, "ce_microbatch_windows": 3})
    assert abs(one[0] - many[0]) < 1e-6 and one[1] == many[1] == 6
    assert ce.adapter_digest(aa) != before
    assert all(torch.allclose(x, y, atol=3e-6, rtol=2e-4) for x, y in zip(aa.parameters(), bb.parameters()))
    parent = Path(__file__).resolve().parent.parent.parent / "lora-ce-preflight-20261119"
    parent.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="cpu-", dir=parent))
    raw = fresh(); prefix = [1, 2]; proposal = []
    for _ in range(3): proposal.append(ce.native_token(raw, prefix + proposal))
    samples = [{"request_id": "fixture", "prompt_ids": prefix,
                "rounds": [{"candidate_ids": proposal, "committed_ids": proposal}]}]
    full, resumed = root / "full", root / "resumed"
    full.mkdir(); resumed.mkdir()
    for directory in (full, resumed):
        torch.manual_seed(42)
        ce.smoke(fresh(), data, cfg, "fixture", directory, samples=samples)
    torch.manual_seed(42)
    ce.train(fresh(), data, calibration, request_count, cfg, "fixture", full)
    original_save = ce.save
    triggered = False
    def interrupted(path, state, adapter, optimizer):
        nonlocal triggered
        original_save(path, state, adapter, optimizer)
        if not triggered and path.name == "resume.pt" and state["updates"] == 1:
            triggered = True
            raise RuntimeError("intentional_CPU_interrupt_after_saved_update")
    ce.save = interrupted
    torch.manual_seed(42)
    try:
        ce.train(fresh(), data, calibration, request_count, cfg, "fixture", resumed)
        raise AssertionError("interruption fixture did not interrupt")
    except RuntimeError as error:
        assert str(error) == "intentional_CPU_interrupt_after_saved_update"
    finally:
        ce.save = original_save
    torch.manual_seed(42)
    ce.train(fresh(), data, calibration, request_count, cfg, "fixture", resumed)
    x = torch.load(full / "last.pt", weights_only=True)
    y = torch.load(resumed / "last.pt", weights_only=True)
    assert x["updates"] == y["updates"] == 4 and x["tokens_seen"] == y["tokens_seen"] == 12
    assert all(torch.equal(x["adapter"][key], y["adapter"][key]) for key in x["adapter"])
    result = {"status": "pass", "checks": ["causal label alignment", "equal/front valid-window CE",
              "unequal microbatches preserve effective-window mean", "zero/internal-LoRA native candidate parity",
              "frozen base/noTarget training", "exact saved-update resume and selected weights",
              "native round audit and tampered acceptance rejection"],
              "scope": "actual tiny Qwen3 CPU; production A100 short smoke still required", "artifacts": str(root)}
    (parent / "cpu_ce_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
