"""Profile manuscript efficiency-table models on one CUDA device.

The script runs one model per process to avoid module-name collisions between
the released Autoformer/iTransformer repositories.  It reports architecture
size, profiler-estimated forward FLOPs, 100 optimization-step time, batch-1
latency after warm-up, and peak allocated CUDA memory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import io
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class CallableWrapper(nn.Module):
    def __init__(self, module: nn.Module, fn):
        super().__init__()
        self.module = module
        self.fn = fn

    def forward(self, x):
        return self.fn(self.module, x)


DATASET_PROTOCOLS = {
    "nasa": {"seq_len": 16, "train_batch": 32},
    "calce": {"seq_len": 64, "train_batch": 128},
    "tju": {"seq_len": 64, "train_batch": 128},
}


def _autoformer(dataset: str):
    seq_len = DATASET_PROTOCOLS[dataset]["seq_len"]
    label_len = seq_len // 2
    source = ROOT / "Compare-Models" / "Autoformer"
    sys.path.insert(0, str(source))
    from models.Autoformer import Model
    cfg = SimpleNamespace(
        task_name="long_term_forecast", seq_len=seq_len, label_len=label_len, pred_len=1,
        enc_in=1, dec_in=1, c_out=1, e_layers=2, d_layers=1, n_heads=4,
        factor=1, d_model=32, d_ff=128,
        moving_avg=5 if seq_len == 16 else 25, dropout=0.1,
        embed="timeF", freq="m", activation="gelu", output_attention=False,
    )
    model = Model(cfg)
    def run(m, x):
        mark = torch.zeros(x.size(0), seq_len, 1, device=x.device)
        dec = torch.cat((x[:, -label_len:], torch.zeros(x.size(0), 1, 1, device=x.device)), 1)
        dec_mark = torch.zeros(x.size(0), label_len + 1, 1, device=x.device)
        return m(x, mark, dec, dec_mark)[:, -1]
    config = dict(seq_len=seq_len, label_len=label_len, pred_len=1, enc_in=1,
                  d_model=32, d_ff=128, e_layers=2, d_layers=1, n_heads=4)
    return CallableWrapper(model, run), (seq_len, 1), \
        "dataset-specific main-experiment reproduction", config


def _itransformer(dataset: str):
    seq_len = DATASET_PROTOCOLS[dataset]["seq_len"]
    source = ROOT / "Compare-Models" / "iTransformer"
    sys.path.insert(0, str(source))
    from model.iTransformer import Model
    cfg = SimpleNamespace(
        task_name="long_term_forecast", seq_len=seq_len, pred_len=1,
        output_attention=False, use_norm=True, d_model=64, embed="timeF",
        freq="m", dropout=0.1, factor=1, n_heads=4, d_ff=128,
        activation="gelu", e_layers=2, class_strategy="projection",
    )
    model = Model(cfg)
    config = dict(seq_len=seq_len, pred_len=1, input_features=4, d_model=64,
                  d_ff=128, e_layers=2, n_heads=4)
    return CallableWrapper(model, lambda m, x: m(x, None, None, None)[:, -1]), (seq_len, 4), \
        "dataset-specific main experiment; four causal capacity-derived variates", config


def _patchformer(dataset: str):
    seq_len = DATASET_PROTOCOLS[dataset]["seq_len"]
    sys.path.insert(0, str(ROOT / "Compare-Models" / "PatchFormer"))
    from ModelsModify.PatchFormer import PatchFormer
    model = PatchFormer(2, seq_len, 1, 1, d_model=16, factor=3, dropout=0.1,
                        output_attention=False, n_heads=8, activation="gelu", e_layers=2)
    config = dict(patch_len=2, seq_len=seq_len, pred_len=1, enc_in=1,
                  d_model=16, factor=3, n_heads=8, e_layers=2)
    return model, (seq_len, 1), \
        "dataset-specific main-experiment reproduction", config


def _rulmamba(dataset: str):
    sys.path.insert(0, str(ROOT / "Compare-Models" / "RUL-Mamba"))
    from Models.RULMamba import RULMamba
    seq_len = DATASET_PROTOCOLS[dataset]["seq_len"]
    if dataset in {"nasa", "calce"}:
        config = dict(enc_in=1, d_model=48, n_dec_layer=1, dropout=0.0615, expand=2)
    else:
        config = dict(enc_in=17, d_model=16, n_dec_layer=2, dropout=0.1, expand=2)
    model = RULMamba(**config)
    return CallableWrapper(model, lambda m, x: m(x, None)), (seq_len, config["enc_in"]), \
        "dataset-specific formal-run architecture", dict(seq_len=seq_len, **config)


def _ic2ml(dataset: str):
    sys.path.insert(0, str(ROOT / "Compare-Models" / "IC2ML"))
    from models.IC2ML_direct import Model
    seq_len = DATASET_PROTOCOLS[dataset]["seq_len"]
    input_dim = 16 if dataset == "tju" else 10
    args = SimpleNamespace(context=seq_len, horizon=1, hidden_dim=256, input_dim=input_dim,
                           use_cycle_input=False, use_capacity_history=True)
    model = Model(args)
    def run(m, x):
        cycles = torch.zeros(x.size(0), x.size(1), device=x.device)
        history = x.mean(-1)
        return m(x, cycles, history)[1]
    config = dict(context=seq_len, horizon=1, hidden_dim=256, input_dim=input_dim,
                  use_cycle_input=False, use_capacity_history=True)
    return CallableWrapper(model, run), (seq_len, input_dim), \
        "dataset-specific formal direct IC2ML with capacity-history input", config


def _batter_moe(dataset: str):
    sys.path.insert(0, str(ROOT / "Compare-Models" / "MOE"))
    from batter_moe.config import get_calce_config, get_paper_config
    from batter_moe.model import BATTERMoE
    if dataset == "calce":
        cfg = get_calce_config("CS2_35").model
    else:
        cfg = get_paper_config(dataset).model
    if dataset == "tju":
        cfg = replace(cfg, use_latest_observation_readout=True)
    model = BATTERMoE(cfg)
    return CallableWrapper(model, lambda m, x: m(x).prediction), \
        (cfg.lookback, cfg.input_channels), \
        "dataset-specific formal-run BATTER-MoE architecture", asdict(cfg)


def _ours(dataset: str):
    from mgi_dssm.physics_model import PhysicsGuidedStateModel
    configs = {
        "nasa": dict(cutoff_voltage_v=2.7, discharge_current_a=2.0,
                     tau_p_seconds=120.0, hidden_dim=48, num_layers=2,
                     dropout=0.0, q_grid_max_ah=2.4, q_grid_points=1200,
                     ocp_profile="lco_graphite", trend_short_window=2,
                     trend_long_window=4),
        "calce": dict(cutoff_voltage_v=2.7, discharge_current_a=1.1,
                      tau_p_seconds=120.0, hidden_dim=32, num_layers=2,
                      dropout=0.0, q_grid_max_ah=1.5, q_grid_points=400,
                      ocp_profile="lco_graphite", trend_short_window=8,
                      trend_long_window=32),
        "tju": dict(cutoff_voltage_v=2.5, discharge_current_a=2.5,
                    tau_p_seconds=120.0, hidden_dim=48, num_layers=1,
                    dropout=0.0, q_grid_max_ah=3.0, q_grid_points=600,
                    ocp_profile="nmc_graphite_siox", trend_short_window=8,
                    trend_long_window=32),
    }
    config = configs[dataset]
    model = PhysicsGuidedStateModel(**config)
    seq_len = DATASET_PROTOCOLS[dataset]["seq_len"]
    return CallableWrapper(model, lambda m, x: m(x)["capacity_ah"]), (seq_len, 5), \
        "formal online state transition plus fixed physical capacity decoder", config


BUILDERS = {
    "PatchFormer": _patchformer,
    "RUL-Mamba": _rulmamba,
    "IC2ML": _ic2ml,
    "BATTER-MoE": _batter_moe,
    "Autoformer": _autoformer,
    "iTransformer": _itransformer,
    "Ours": _ours,
}


def output_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        return sum((output_tensor(item).float().sum() for item in value), torch.tensor(0.0, device=value[0].device))
    raise TypeError(type(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(BUILDERS), required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_PROTOCOLS), required=True)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--train-batch", type=int, default=None)
    parser.add_argument("--measurement-rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    model, shape, basis, architecture_config = BUILDERS[args.model](args.dataset)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_size_mb = len(buffer.getvalue()) / (1024 ** 2)
    model = model.to(device)
    model.eval()
    x1 = torch.randn((1,) + shape, device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x1)
    torch.cuda.synchronize()
    latencies = []
    with torch.inference_mode():
        for _ in range(args.repeats):
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)
            start.record(); model(x1); end.record(); end.synchronize()
            latencies.append(float(start.elapsed_time(end)))

    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        model(x1)
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # torch.profiler counts supported matrix/conv operations; unsupported
    # element-wise operations are conservatively absent and this is disclosed.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_flops=True,
    ) as prof:
        with torch.inference_mode():
            model(x1)
        torch.cuda.synchronize()
    flops = sum(event.flops for event in prof.key_averages() if event.flops)

    model.train()
    train_batch = args.train_batch or DATASET_PROTOCOLS[args.dataset]["train_batch"]
    xb = torch.randn((train_batch,) + shape, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = output_tensor(model(xb)).float().mean()
        loss.backward(); optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    training_rounds = []
    for _ in range(args.measurement_rounds):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(args.train_steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = output_tensor(model(xb)).float()
            loss = prediction.square().mean()
            loss.backward(); optimizer.step()
        torch.cuda.synchronize()
        training_rounds.append(time.perf_counter() - start_time)
    training_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    result = {
        "model": args.model,
        "dataset": args.dataset,
        "parameter_basis": basis,
        "architecture_config": architecture_config,
        "params": params,
        "params_k": params / 1000.0,
        "flops": flops,
        "flops_m": flops / 1e6,
        "model_size_mb": model_size_mb,
        "training_time_100_steps_s": statistics.median(training_rounds) * 100 / args.train_steps,
        "training_time_rounds_s": training_rounds,
        "inference_median_ms": statistics.median(latencies),
        "inference_iqr_ms": statistics.quantiles(latencies, n=4)[2] - statistics.quantiles(latencies, n=4)[0],
        "inference_peak_memory_mb": peak_mb,
        "training_peak_memory_mb": training_peak_mb,
        "input_shape_batch1": [1, *shape],
        "warmup": args.warmup,
        "repeats": args.repeats,
        "train_batch": train_batch,
        "measurement_rounds": args.measurement_rounds,
        "precision": "FP32",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
