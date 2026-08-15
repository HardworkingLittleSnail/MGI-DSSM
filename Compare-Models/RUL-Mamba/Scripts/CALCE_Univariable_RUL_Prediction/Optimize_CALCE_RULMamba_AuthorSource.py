"""Validation-only TPE search for the parent author RUL-Mamba source on CALCE.

CS2_35 is held out from every hyperparameter trial. It is evaluated exactly
once after the best validation configuration has been selected.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path

import optuna

import Train_CALCE_Univariable as train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--battery", default="CS2_35")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--trial-epochs", type=int, default=80)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    deps = train.load_dependencies()
    train.setup_legacy_module_aliases()
    deps.StableSMAPE = train.import_object("Models.Metrics.StableSMAPE")
    base = train.deep_merge(train.load_yaml(args.config), train.load_yaml(args.model_config))
    base["train"]["seeds"] = [args.seed]
    base["protocol"]["preprocessing"] = "raw"
    base["protocol"]["normalization_scope"] = "train_prefix"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    batteries = train.load_calce_cache(
        train.PROJECT_ROOT / base["dataset"]["battery_cache_path"],
        base["dataset"]["battery_list"],
    )
    batteries, preprocessing = train.preprocess_calce_batteries(batteries, "raw")
    start_point = int(base["dataset"]["start_points"][args.battery])
    train_frame, val_frame, _, protocol = train.prepare_fold(
        batteries=batteries,
        test_name=args.battery,
        start_point=start_point,
        seq_len=base["window"]["seq_len"],
        validation_fraction=base["train"]["validation_fraction"],
        rated_capacity=base["dataset"]["rated_capacity"],
        normalization_scope="train_prefix",
    )
    model_class = train.import_object(base["model"]["class_path"])

    def objective(trial: optuna.Trial) -> float:
        gc.collect()
        if deps.torch.cuda.is_available():
            deps.torch.cuda.empty_cache()
        config = copy.deepcopy(base)
        requested_batch = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        config["train"]["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-4, 1e-2, log=True
        )
        config["train"]["gradient_clip_val"] = trial.suggest_float(
            "gradient_clip_val", 0.1, 0.5, log=True
        )
        config["train"]["drop_last_train"] = True
        build = config["model"]["build_args"]
        build["d_model"] = trial.suggest_int("d_model", 8, 128, step=8)
        build["n_dec_layer"] = trial.suggest_int("n_dec_layer", 1, 3)
        build["dropout"] = trial.suggest_float("dropout", 0.01, 0.2)
        # The author's Python selective scan stores O(B*L*d_inner*d_state)
        # intermediates. Keep all paper-valid dimensions searchable while
        # capping only the effective batch to the 12-GB hardware envelope.
        max_safe_batch = 32 if build["d_model"] >= 80 else 64
        config["train"]["batch_size"] = min(requested_batch, max_safe_batch)
        trial.set_user_attr("effective_batch_size", config["train"]["batch_size"])

        train.set_seed(args.seed + trial.number, deps.torch)
        training = train.build_dataset(train_frame, config, deps)
        validating = train.build_dataset(val_frame, config, deps)
        train_loader = training.to_dataloader(
            train=True,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        val_loader = validating.to_dataloader(
            train=False,
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        model = model_class.from_dataset(
            training,
            seq_len=build["seq_len"], pred_len=build["pred_len"],
            enc_in=build["enc_in"], c_out=build["c_out"],
            d_model=build["d_model"], n_dec_layer=build["n_dec_layer"],
            dropout=build["dropout"], expand=build["expand"],
            learning_rate=config["train"]["learning_rate"],
            weight_decay=0.0, optimizer="adam", loss=deps.StableSMAPE(),
            logging_metrics=deps.torch.nn.ModuleList([deps.MAE()]),
            reduce_on_plateau_patience=1000,
        )
        early = deps.EarlyStopping(
            monitor="val_loss", min_delta=1e-5, patience=10, mode="min"
        )
        checkpoint = deps.ModelCheckpoint(
            dirpath=str(output / "trials" / f"trial_{trial.number}"),
            monitor="val_loss", mode="min", save_top_k=1,
            filename="{epoch:02d}-{val_loss:.8f}",
        )
        trainer = deps.pl.Trainer(
            max_epochs=args.trial_epochs,
            gradient_clip_val=config["train"]["gradient_clip_val"],
            callbacks=[early, checkpoint], logger=False,
            default_root_dir=str(output / "trials" / f"trial_{trial.number}"),
            accelerator="gpu" if deps.torch.cuda.is_available() else "cpu",
            devices=1, deterministic=True, enable_progress_bar=False,
            enable_model_summary=False, num_sanity_val_steps=0,
        )
        try:
            with deps.torch.enable_grad():
                trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            value = float(checkpoint.best_model_score.detach().cpu())
            trial.set_user_attr("best_epoch", int(
                deps.torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)["epoch"]
            ))
            trial.set_user_attr("checkpoint", checkpoint.best_model_path)
            return value
        except deps.torch.cuda.OutOfMemoryError:
            trial.set_user_attr("failure", "cuda_out_of_memory")
            raise
        finally:
            del trainer, model, train_loader, val_loader, training, validating
            gc.collect()
            if deps.torch.cuda.is_available():
                deps.torch.cuda.empty_cache()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        storage=f"sqlite:///{(output / 'study.sqlite3').as_posix()}",
        study_name="calce_rulmamba_author_source",
        load_if_exists=True,
    )
    remaining = max(0, args.trials - len(study.trials))
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            catch=(deps.torch.cuda.OutOfMemoryError,),
        )

    best = study.best_trial
    final_config = copy.deepcopy(base)
    final_config["train"].update({
        "seeds": [args.seed],
        "batch_size": best.user_attrs.get("effective_batch_size", best.params["batch_size"]),
        "learning_rate": best.params["learning_rate"],
        "gradient_clip_val": best.params["gradient_clip_val"],
        "drop_last_train": True,
        "max_epochs": 200,
        "patience": 20,
    })
    final_config["model"]["build_args"].update({
        "d_model": best.params["d_model"],
        "n_dec_layer": best.params["n_dec_layer"],
        "dropout": best.params["dropout"],
    })
    final_config["output"] = {
        "outputs_dir": str(output / "final"),
        "logs_dir": str(output / "final_logs"),
    }
    final_config["protocol"]["preprocessing"] = "raw"
    final_config["protocol"]["normalization_scope"] = "train_prefix"
    train.save_yaml(final_config, output / "best_config.yaml")
    summary = {
        "best_trial": best.number,
        "best_validation_smape": best.value,
        "best_params": best.params,
        "best_epoch_in_search": best.user_attrs.get("best_epoch"),
        "trials": len(study.trials),
        "held_out_battery_not_used_by_hpo": args.battery,
        "preprocessing": preprocessing,
        "protocol": protocol,
    }
    train.save_json(summary, output / "hpo_summary.json")
    print(json.dumps(summary, indent=2), flush=True)

    # Evaluate the held-out battery once, after validation-only selection.
    final_root = Path(final_config["output"]["outputs_dir"])
    final_logs = Path(final_config["output"]["logs_dir"])
    metrics = train.run_one(
        args.battery, args.seed, batteries, final_config,
        final_root, final_logs, model_class, deps, True,
    )
    train.save_json(metrics, output / "final_test_metrics.json")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
