"""Direct Draft LoRA with saved RAW5000 target tokens; no teacher collection."""

import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from draft_interventions import attach_lora, enable_lora
from program62_data import atomic_json, checked_file, digest_file, jsonl, read_json, require
from program62_compare import compare_by_family, read_eval
import program62_run as runtime


HERE = Path(__file__).resolve().parent


def windows(records, split):
    result = []
    requests = [row for row in records if row["split"] == split]
    for request_index, record in enumerate(requests):
        require(record["prompt_ids"] and record["output_ids"] and record["windows"], "empty frozen request")
        for row in record["windows"]:
            offset, labels, count = row["offset"], row["labels"], row["length"]
            require(0 <= offset and 1 <= count <= 8 and len(labels) == count
                    and labels == record["output_ids"][offset:offset + count], "saved Target labels differ")
            result.append((request_index, record, row))
    return result, len(requests)


def sequence(window):
    _, record, row = window
    prefix = record["prompt_ids"] + record["output_ids"][:row["offset"]]
    return prefix + row["labels"][:-1], len(prefix) - 1, row["labels"]


def microbatches(batch, cfg):
    current, longest = [], 0
    for window in batch:
        length = len(sequence(window)[0])
        candidate_length = max(length, longest)
        count = len(current) + 1
        fits = (count <= cfg["ce_microbatch_windows"]
                and count * candidate_length <= cfg["ce_microbatch_tokens"]
                and count * candidate_length ** 2 <= cfg["ce_microbatch_attention_cells"])
        if current and not fits:
            yield current
            current, longest = [], 0
        current.append(window)
        longest = max(longest, length)
    if current:
        yield current


def losses(model, batch, device):
    """Right-pad actual prefixes, score only the existing <=8 target labels."""
    data = [sequence(window) for window in batch]
    longest = max(len(ids) for ids, _, _ in data)
    pad = model.config.pad_token_id or 0
    ids = torch.full((len(data), longest), pad, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    for i, (tokens, _, _) in enumerate(data):
        ids[i, :len(tokens)] = torch.tensor(tokens, device=device)
        attention[i, :len(tokens)] = 1
    hidden = model.model(input_ids=ids, attention_mask=attention,
                         use_cache=False, return_dict=True).last_hidden_state
    selected = torch.cat([hidden[i, position:position + len(labels)]
                          for i, (_, position, labels) in enumerate(data)])
    labels = torch.tensor([token for _, _, values in data for token in values], device=device)
    per_token = F.cross_entropy(model.get_output_embeddings()(selected).float(), labels, reduction="none")
    equal, front, offset = [], [], 0
    for _, _, values in data:
        count = len(values)
        local = per_token[offset:offset + count]
        weight = torch.arange(8, 8 - count, -1, device=device, dtype=torch.float32)
        equal.append(local.mean())
        front.append((local * weight).sum() / weight.sum())
        offset += count
    return {"equal": torch.stack(equal), "front": torch.stack(front)}, len(labels)


def update(model, adapter, batch, optimizer, cfg):
    optimizer.zero_grad(set_to_none=True)
    total_loss, tokens = 0., 0
    for micro in microbatches(batch, cfg):
        values, count = losses(model, micro, model.device)
        loss = values[cfg["weighting"]].sum() / len(batch)
        require(bool(loss.isfinite()), "nonfinite saved-token CE")
        loss.backward()
        total_loss += float(loss.detach())
        tokens += count
    # Match the existing first-four heads: observe finite norm without clipping.
    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), float("inf"), error_if_nonfinite=True)
    optimizer.step()
    return total_loss, tokens, float(norm)


def adapter_digest(adapter):
    value = sha256()
    for name, tensor in sorted(adapter.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def optimizer_for(adapter, cfg):
    return torch.optim.AdamW(adapter.parameters(), lr=cfg["learning_rate"],
                             betas=(.9, .999), eps=1e-8, weight_decay=cfg["weight_decay"])


def load_draft(cfg):
    """Training and smoke load only the 0.6B Draft, never the Target."""
    checked_file(cfg["model_manifest"], cfg["model_manifest_sha256"])
    manifest = read_json(cfg["model_manifest"])
    root = Path(cfg["models_root"]) / "Qwen3-0.6B"
    for name, entry in manifest["Qwen3-0.6B"].items():
        checked_file(root / name, entry["sha256"])
    sys.path.insert(0, cfg["source_root"])
    checked_file(Path(cfg["source_root"]) / "feedback_codec/hf_backend.py", cfg["backend_sha256"])
    from feedback_codec.hf_backend import _FrozenHFBackend
    return _FrozenHFBackend(str(root), None, device=cfg["device"], dtype=torch.bfloat16)


def native_token(draft, prefix):
    output, _ = draft._forward_with_final_hidden(draft._ids(prefix))
    return int(output.logits[0, -1].argmax())


def smoke(draft, train, cfg, cfg_sha, root, samples=None):
    adapter = attach_lora(draft.model, cfg["lora_rank"], cfg["lora_alpha"])
    initial = adapter_digest(adapter)
    samples = read_json(HERE / "inputs/native_raw503_samples.json") if samples is None else samples
    checked_candidates = 0
    with torch.inference_mode():
        for request in samples:
            prefix = request["prompt_ids"].copy()
            for row in request["rounds"]:
                proposed = []
                for expected in row["candidate_ids"]:
                    enable_lora(draft.model, False)
                    raw = native_token(draft, prefix + proposed)
                    enable_lora(draft.model, True)
                    zero = native_token(draft, prefix + proposed)
                    require(raw == zero == expected, "native Draft/zero LoRA differs from sealed RAW candidate")
                    proposed.append(zero)
                    checked_candidates += 1
                prefix.extend(row["committed_ids"])
    frozen = {name: value._version for name, value in draft.model.named_parameters() if not value.requires_grad}
    batch = train[:cfg["ce_batch_windows"]]
    loss, tokens, norm = update(draft.model, adapter, batch, optimizer_for(adapter, cfg), cfg)
    require(norm > 0 and adapter_digest(adapter) != initial, "LoRA did not change")
    require(all(value.grad is None and value._version == frozen[name]
                for name, value in draft.model.named_parameters() if name in frozen), "Draft base changed in LoRA smoke")
    atomic_json(root / "smoke.json", {"status": "pass", "config_sha256": cfg_sha,
                "arm": "lora_ce", "initial_adapter_sha256": initial, "CE": loss,
                "gradient_norm": norm, "training_windows": len(batch), "valid_tokens": tokens,
                "cached_RAW_requests": len(samples), "cached_RAW_candidate_tokens_checked": checked_candidates,
                "Target_loaded": False, "new_teacher_labels": 0})


def score(model, data, request_count, cfg):
    sums = {name: [0.] * request_count for name in ("equal", "front")}
    counts = [0] * request_count
    with torch.no_grad():
        for micro in microbatches(data, cfg):
            values, _ = losses(model, micro, model.device)
            for i, window in enumerate(micro):
                request_index = window[0]
                counts[request_index] += 1
                for name in sums:
                    sums[name][request_index] += float(values[name][i])
    require(all(counts), "calibration request has no windows")
    return {name: sum(v / n for v, n in zip(values, counts)) / request_count for name, values in sums.items()}


def save(path, state, adapter, optimizer):
    pending = path.with_suffix(".pending")
    torch.save({**state, "adapter": adapter.state_dict(), "optimizer": optimizer.state_dict()}, pending)
    pending.replace(path)


def train(draft, data, calibration, calibration_requests, cfg, cfg_sha, root):
    terminal = root / "completion.json"
    if terminal.exists():
        result = read_json(terminal)
        require(result["config_sha256"] == cfg_sha and result["status"] == "complete", "training terminal differs")
        checked_file(root / "last.pt", result["checkpoint_sha256"])
        return
    preflight = read_json(root / "smoke.json")
    require(preflight["status"] == "pass" and preflight["config_sha256"] == cfg_sha, "same-config smoke missing")
    adapter = attach_lora(draft.model, cfg["lora_rank"], cfg["lora_alpha"])
    initial = adapter_digest(adapter)
    require(initial == preflight["initial_adapter_sha256"], "training LoRA initialization differs from smoke")
    optimizer = optimizer_for(adapter, cfg)
    state = {"arm": "lora", "config_sha256": cfg_sha, "epoch": 1, "next_position": 0,
             "updates": 0, "tokens_seen": 0, "history": [], "initial_adapter_sha256": initial}
    resume = root / "resume.pt"
    if resume.exists():
        old = torch.load(resume, map_location="cpu", weights_only=True)
        require(old["config_sha256"] == cfg_sha and old["initial_adapter_sha256"] == initial, "training resume differs")
        adapter.load_state_dict(old.pop("adapter"))
        optimizer.load_state_dict(old.pop("optimizer"))
        state = old
    started = time.time()
    first_epoch = state["epoch"]
    for epoch in range(first_epoch, cfg["ce_epochs"] + 1):
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(cfg["seed"] + epoch)).tolist()
        position = state["next_position"] if epoch == first_epoch else 0
        for offset in range(position, len(order), cfg["ce_batch_windows"]):
            batch = [data[i] for i in order[offset:offset + cfg["ce_batch_windows"]]]
            loss, tokens, norm = update(draft.model, adapter, batch, optimizer, cfg)
            state.update(epoch=epoch, next_position=offset + len(batch), updates=state["updates"] + 1,
                         tokens_seen=state["tokens_seen"] + tokens)
            if state["updates"] == 1 or state["updates"] % cfg["checkpoint_interval"] == 0:
                save(resume, state, adapter, optimizer)
                atomic_json(root / "status.json", {"phase": "train", "epoch": epoch, "epoch_windows": state["next_position"],
                            "epoch_total": len(data), "updates": state["updates"], "last_CE": loss,
                            "gradient_norm": norm, "tokens_seen": state["tokens_seen"],
                            "elapsed_this_process_seconds": time.time() - started, "Target_loaded": False})
        save(resume, state, adapter, optimizer)
        calibration_ce = score(draft.model, calibration, calibration_requests, cfg)
        state["history"].append({"epoch": epoch, "updates": state["updates"], "calibration_request_equal_CE": calibration_ce})
        epoch_path = root / f"epoch{epoch}.pt"
        save(epoch_path, state, adapter, optimizer)
        selected_path = root / "selected.json"
        selected = read_json(selected_path) if selected_path.exists() else None
        metric = calibration_ce[cfg["weighting"]]
        if selected is None or metric < selected["calibration_CE"]:
            atomic_json(selected_path, {"epoch": epoch, "path": str(epoch_path), "sha256": digest_file(epoch_path),
                        "calibration_CE": metric, "criterion": cfg["weighting"], "config_sha256": cfg_sha})
        state.update(epoch=epoch + 1, next_position=0)
        save(resume, state, adapter, optimizer)
        atomic_json(root / "history.json", state["history"])
    require(state["updates"] == cfg["ce_epochs"] * ((len(data) + cfg["ce_batch_windows"] - 1) // cfg["ce_batch_windows"]), "update budget differs")
    require(state["tokens_seen"] == cfg["ce_epochs"] * sum(len(sequence(window)[2]) for window in data),
            "saved supervision token budget differs")
    selected = read_json(root / "selected.json")
    checked_file(selected["path"], selected["sha256"])
    shutil.copyfile(selected["path"], root / "last.pt")
    atomic_json(terminal, {**state, "status": "complete", "selected_epoch": selected["epoch"],
                "checkpoint_sha256": digest_file(root / "last.pt"), "new_teacher_labels": 0,
                "Target_loaded_during_training": False, "training_windows": len(data)})


def native_request_eval(cfg, decode, target, draft, request, slot=None, mode="raw", donors=None):
    require(slot is None and mode in ("raw", "lora", "mask"), "CE arm must directly change Draft")
    first = None
    def token(prefix, state, position):
        return draft.next(prefix, None, position)
    def target_step(prefix):
        step = target.next(prefix)
        return SimpleNamespace(token_id=step.token_id, hidden=step.hidden, source_prefix=tuple(prefix))
    def observed(row, boundary):
        nonlocal first
        if first is None:
            if draft.device.type == "cuda": torch.cuda.synchronize(draft.device)
            first = time.perf_counter()
    if draft.device.type == "cuda": torch.cuda.synchronize(draft.device)
    started = time.perf_counter()
    with torch.inference_mode():
        result = decode(request["prompt_token_ids"], draft_token=token, target_step=target_step,
                        prepare=None, on_round=observed, oracle_ids=request["expected_target_token_ids"],
                        max_new=cfg["max_new"], window=cfg["window"], eos=cfg["eos_token_id"])
    if draft.device.type == "cuda": torch.cuda.synchronize(draft.device)
    result.update(wall_seconds=time.perf_counter() - started, first_commit_seconds=first - started,
                  request_id=request["request_id"], family_id=request["family_id"], mode=mode)
    return result


def compare(cfg, cfg_sha, root):
    roster = {r["request_id"]: r for r in jsonl(cfg["development_roster"])}
    rows, stats = read_eval(root / "development-lora", cfg_sha, roster, "lora", "development", "lora")
    terminal = read_json(root / "completion.json")
    require(terminal["status"] == "complete" and terminal["config_sha256"] == cfg_sha
            and terminal["arm"] == "lora" and stats["checkpoint_sha256"] == terminal["checkpoint_sha256"],
            "evaluation checkpoint differs from selected training checkpoint")
    checked_file(root / "last.pt", terminal["checkpoint_sha256"])
    for rid, record in rows.items():
        validate_rounds(record, roster[rid]["expected_target_token_ids"], cfg)
    candidate = {rid: (row["accepted_draft_tokens"], row["verification_rounds"]) for rid, row in rows.items()}
    controls = ("RAW", "mlp1_w256_" + cfg["weighting"], "transformer1_w64_" + cfg["weighting"])
    reports = {}
    for name in controls:
        control = {rid: tuple(row["baseline_counts"][name]) for rid, row in roster.items()}
        reports[name] = compare_by_family(roster, control, candidate, cfg["seed"], 10000)
    atomic_json(root / "comparison.json", {"status": "complete", "objective": "saved_token_CE",
                "config_sha256": cfg_sha, "comparisons": reports, "candidate_statistics": stats,
                "scope": "503 development requests; old RAW/first4 head results reused from sealed source. No fresh RAW timing comparison or independent speed claim."})


def validate_rounds(record, expected, cfg):
    cursor = accepted_total = 0
    for row in record["rounds"]:
        proposal = row["candidate_ids"]
        require(proposal and len(proposal) <= min(cfg["window"], cfg["max_new"] - cursor)
                and cfg["eos_token_id"] not in proposal[:-1], "invalid proposal horizon")
        accepted = 0
        for token in proposal:
            if cursor + accepted >= len(expected) or token != expected[cursor + accepted]: break
            accepted += 1
            if token == cfg["eos_token_id"]: break
        require(accepted == row["accepted_count"], "round accepted count differs from actual proposal")
        terminal = accepted > 0 and (proposal[accepted - 1] == cfg["eos_token_id"] or cursor + accepted == cfg["max_new"])
        correction_or_bonus = accepted < len(proposal) or not terminal
        committed = row["committed_ids"]
        require(len(committed) == accepted + int(correction_or_bonus)
                and committed == expected[cursor:cursor + len(committed)]
                and row["target_decision_ids"] == committed, "round Target or commit differs")
        cursor += len(committed)
        accepted_total += accepted
    require(cursor == len(expected) and record["output_ids"] == expected
            and accepted_total == record["accepted_draft_tokens"]
            and record["actual_target_forwards"] == len(expected), "round/output totals differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("configure", "smoke", "train", "eval", "compare"))
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--loss", choices=("equal", "front"), required=True)
    parser.add_argument("--gpu-lock", type=Path, default=Path("/tmp/spec-experiments.cuda0.lock"))
    args = parser.parse_args()
    cfg = read_json(HERE / "ce_candidate.json")
    for name, expected in cfg["implementation_sha256"].items(): checked_file(HERE / name, expected)
    cfg.update(source_root=str(HERE / "vendor"), decoder=str(HERE / "vendor/closed_loop_core.py"),
               model_manifest=str(args.assets.resolve() / "models.json"), models_root=str(args.assets.resolve() / "weights"),
               train_index=str(args.assets.resolve() / "train/index.jsonl"), gpu_lock=str(args.gpu_lock.resolve()),
               development_roster=str(HERE / "inputs/development503_existing.jsonl"), weighting=args.loss)
    checked_file(cfg["development_roster"], cfg["development_roster_sha256"])
    checked_file(HERE / "inputs/native_raw503_samples.json", cfg["raw_samples_sha256"])
    root = args.out.resolve() / args.loss / "lora"
    require(not root.is_relative_to(HERE), "outputs must be outside checkout")
    path = root.parent / "config.json"
    if path.exists(): require(read_json(path) == cfg, "existing output config differs")
    else: atomic_json(path, cfg)
    cfg_sha = digest_file(path)
    if args.phase == "configure": print(path); return
    if args.phase == "compare": compare(cfg, cfg_sha, root); return
    torch.manual_seed(cfg["seed"])
    with runtime.worker_lease(cfg, root):
        if args.phase in ("smoke", "train"):
            checked_file(cfg["train_index"], cfg["train_index_sha256"])
            records = list(jsonl(cfg["train_index"]))
            data, train_requests = windows(records, "train")
            calibration, calibration_requests = windows(records, "calibration")
            require((len(data), train_requests, len(calibration), calibration_requests)
                    == (cfg["train_windows"], 3986, cfg["calibration_windows"], 503), "existing dataset counts differ")
            draft = load_draft(cfg)
            if args.phase == "smoke": smoke(draft, data, cfg, cfg_sha, root)
            else: train(draft, data, calibration, calibration_requests, cfg, cfg_sha, root)
        else:
            terminal = read_json(root / "completion.json")
            require(terminal["status"] == "complete" and terminal["config_sha256"] == cfg_sha, "training terminal absent")
            checked_file(root / "last.pt", terminal["checkpoint_sha256"])
            target, draft = runtime.load_models(cfg)
            runtime.request_eval = native_request_eval
            runtime.evaluate(cfg, cfg_sha, "lora", "lora", target, draft,
                             root / "development-lora", root / "last.pt", "development")


if __name__ == "__main__":
    main()
