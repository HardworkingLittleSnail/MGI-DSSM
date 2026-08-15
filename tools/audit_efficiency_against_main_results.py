"""Cross-check efficiency builders against the exact formal-run configurations."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EFFICIENCY = ROOT / "outputs" / "efficiency_three_datasets_rtx3080ti"
FORMAL = ROOT / "outputs" / "seven_models_version3_10seeds"
TRANSFORMERS = ROOT / "outputs" / "autoformer_itransformer_official_version3_10seeds_200ep"
BATTERIES = {"nasa": "B0005", "calce": "CS2_35", "tju": "CY25-1"}
MODELS = (
    "PatchFormer", "RUL-Mamba", "IC2ML", "BATTER-MoE",
    "Autoformer", "iTransformer", "Ours",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def formal_result(dataset: str, model: str) -> dict:
    battery = BATTERIES[dataset]
    if model in {"Autoformer", "iTransformer"}:
        folder = model.lower()
        return load(TRANSFORMERS / folder / dataset / battery / "seed_7" / "results.json")
    folder = {
        "PatchFormer": "patchformer", "RUL-Mamba": "rul-mamba",
        "IC2ML": "ic2ml", "BATTER-MoE": "batter-moe", "Ours": "our_model",
    }[model]
    return load(FORMAL / folder / dataset / battery / "seed_7" / "results.json")


def expected_architecture(dataset: str, model: str, result: dict) -> dict:
    if model == "Ours":
        return result["config"]
    if model in {"PatchFormer", "RUL-Mamba", "Autoformer", "iTransformer"}:
        expected = dict(result["native_config"]["build"])
        if model == "iTransformer":
            expected["input_features"] = len(result["native_config"]["input_features"])
        return expected
    if model == "BATTER-MoE":
        return result["paper_config"]["model"]
    config = result["config"]
    return {
        "context": config["seq_len"], "horizon": 1,
        "hidden_dim": config["hidden_dim"], "input_dim": config["input_dim"],
        "use_cycle_input": config["use_cycle_input"],
        "use_capacity_history": config["use_capacity_history"],
    }


def main() -> None:
    failures: list[str] = []
    rows: list[dict] = []
    for dataset in ("nasa", "calce", "tju"):
        for model in MODELS:
            slug = model.lower().replace("-", "_")
            measured = load(EFFICIENCY / dataset / f"{slug}.json")
            expected = expected_architecture(dataset, model, formal_result(dataset, model))
            actual = measured["architecture_config"]
            mismatches = {
                key: {"measured": value, "formal": expected.get(key)}
                for key, value in actual.items() if expected.get(key) != value
            }
            if mismatches:
                failures.append(f"{dataset}/{model}: {mismatches}")
            rows.append({
                "dataset": dataset, "model": model, "params": measured["params"],
                "input_shape": measured["input_shape_batch1"],
                "architecture_match": not mismatches,
            })

    # Independent source-scale checks.  PatchFormer reports 97.9 K for its
    # L=64 author configuration; BATTER-MoE reports about 94.5 K (NASA) and
    # 5.4 M (TJU).  These are rounded paper/repository values.
    source_checks = [
        ("calce", "PatchFormer", 97_900, 1_000),
        ("tju", "PatchFormer", 97_900, 1_000),
        ("nasa", "BATTER-MoE", 94_500, 1_000),
        ("tju", "BATTER-MoE", 5_400_000, 60_000),
    ]
    for dataset, model, reference, tolerance in source_checks:
        row = next(item for item in rows if item["dataset"] == dataset and item["model"] == model)
        if abs(int(row["params"]) - reference) > tolerance:
            failures.append(
                f"source-scale mismatch {dataset}/{model}: {row['params']} vs {reference}"
            )
    report = {"status": "pass" if not failures else "fail", "rows": rows, "failures": failures}
    (EFFICIENCY / "architecture_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures:
        raise SystemExit("\n".join(failures))
    print("PASS: all 21 efficiency builders match their formal-run configurations")


if __name__ == "__main__":
    main()
