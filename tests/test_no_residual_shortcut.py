from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_comparison_runners_do_not_reconstruct_from_previous_capacity():
    forbidden = (
        "capacity_residual_scale",
        "_capacity_prediction",
        "last_capacity +",
        "anchor + final",
        "increment_mean",
        "increment_std",
    )
    for relative in (
        "Compare-Models/run_physics_dual_loss.py",
        "Compare-Models/run_sg_dits.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern not in source, f"forbidden residual shortcut {pattern!r} in {relative}"

