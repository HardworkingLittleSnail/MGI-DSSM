from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run repeated, independently seeded MSTEA-protocol experiments."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    specification = json.loads(args.config.read_text(encoding="utf-8-sig"))
    batteries = [str(value) for value in specification["batteries"]]
    seeds = [int(value) for value in specification["seeds"]]
    output_root = Path(specification["output_root"])
    base_train = dict(specification["train"])
    start_points_by_battery = {
        str(key): [int(value) for value in values]
        for key, values in specification.get("start_points_by_battery", {}).items()
    }
    generated_config_dir = output_root / "configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    total = len(batteries) * len(seeds)
    completed = 0
    for battery in batteries:
        for run_index, seed in enumerate(seeds, start=1):
            completed += 1
            run_name = f"{battery}_run_{run_index:02d}_seed_{seed}"
            run_dir = output_root / battery / f"run_{run_index:02d}_seed_{seed}"
            run_config = dict(base_train)
            run_config.update(
                {
                    "test_names": [battery],
                    "seed": seed,
                    "output_dir": str(run_dir),
                }
            )
            if battery in start_points_by_battery:
                run_config["start_points"] = start_points_by_battery[battery]
            config_path = generated_config_dir / f"{run_name}.json"
            config_path.write_text(
                json.dumps(run_config, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train.log"
            command = [
                sys.executable,
                "run_mgi_dssm.py",
                "train",
                "--config",
                str(config_path),
            ]
            result_path = run_dir / "results.json"
            if result_path.exists():
                try:
                    json.loads(result_path.read_text(encoding="utf-8"))
                    print(f"\n[{completed}/{total}] skip completed: {result_path}", flush=True)
                except (json.JSONDecodeError, OSError):
                    result_path.unlink(missing_ok=True)
            if not result_path.exists():
                print(f"\n[{completed}/{total}] {' '.join(command)}", flush=True)
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )
                    assert process.stdout is not None
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    return_code = process.wait()
                if return_code != 0:
                    raise RuntimeError(
                        f"Experiment failed: {run_name}. See {log_path}"
                    )

            payload = json.loads(result_path.read_text(encoding="utf-8"))
            for fold in payload["folds"]:
                summary_rows.append(
                    {
                        "battery": battery,
                        "run": run_index,
                        "seed": seed,
                        "start_point": fold["start_point"],
                        "windows": fold["num_windows"],
                        "mae": fold["mae"],
                        "rmse": fold["rmse"],
                        "r2": fold["r2"],
                        "ecycle": fold["AE"],
                        "raw_ecycle": fold.get("raw_AE", fold["AE"]),
                        "re": fold["RE"],
                        "raw_re": fold.get("raw_RE", fold["RE"]),
                        "eol_phase_delta": fold.get("eol_phase_delta", 0),
                        "persistence_mae": fold["persistence_mae"],
                        "result_path": str(result_path),
                    }
                )
            (output_root / "all_results.json").write_text(
                json.dumps(summary_rows, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            with (output_root / "all_results.csv").open(
                "w", newline="", encoding="utf-8-sig"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
                writer.writeheader()
                writer.writerows(summary_rows)

    print(
        f"\nCompleted {total} experiments. Summary: "
        f"{output_root / 'all_results.csv'}"
    )


if __name__ == "__main__":
    main()
