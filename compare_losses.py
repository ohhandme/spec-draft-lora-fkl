"""Compare front-weighted versus equal-weight Draft-LoRA on the frozen roster."""

import argparse
from pathlib import Path

from program62_compare import compare_by_family, read_eval
from program62_data import atomic_json, checked_file, digest_file, jsonl, read_json, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out.resolve()
    configs = {loss: out / loss / "config.json" for loss in ("equal", "front")}
    loaded = {loss: read_json(path) for loss, path in configs.items()}
    for loss, cfg in loaded.items():
        require(cfg.pop("weighting") == loss, "incorrect loss variant")
    require(loaded["equal"] == loaded["front"], "configs differ in more than the loss weights")
    cfg = loaded["equal"]
    checked_file(cfg["development_roster"], cfg["development_roster_sha256"])
    records = list(jsonl(cfg["development_roster"]))
    roster = {row["request_id"]: row for row in records}
    require(len(roster) == len(records) == 200, "development roster differs")
    summaries, counts = {}, {}
    for loss, path in configs.items():
        rows, summaries[loss] = read_eval(out / loss / "lora" / "development-lora", digest_file(path),
                                         roster, "lora", "development", "lora")
        counts[loss] = {key: (row["accepted_draft_tokens"], row["verification_rounds"])
                        for key, row in rows.items()}
    result = {"comparison": "front minus equal", "summaries": summaries,
              "paired": compare_by_family(roster, counts["equal"], counts["front"], 42, 10000)}
    atomic_json(out / "front-minus-equal.json", result)


if __name__ == "__main__":
    main()
