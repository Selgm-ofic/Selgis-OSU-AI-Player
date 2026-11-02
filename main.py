import argparse
from pathlib import Path
import yaml
from osu2.data import build_dataset
from osu2.train import train
from osu2.benchmark import benchmark
from osu2.evaluate import evaluate
from osu2.preview import preview
from osu2.recording_gui import run as run_gui
from osu2.fast_label import label_session
from osu2.rl_lab import run as run_rl


def main():
    root=Path(__file__).resolve().parent; cfg=yaml.safe_load((root/"config.yaml").read_text())
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    b=sub.add_parser("build"); b.add_argument("--source",default=r"C:\Gamma\Osu_module\data\recordings"); b.add_argument("--out",default=str(root/"data/dataset_v7"))
    t = sub.add_parser("train")
    t.add_argument("--version", default="v8_5", help="Target model version (default: v8_5)")
    t.add_argument("--name", default=None, help="Explicit full model name (e.g. Semga-OSU-0.1-SC-SJA-FC-EXP)")
    t.add_argument("--method", default="two_stage_freeze", choices=["sc_sja_fc", "standard", "two_stage_freeze"], help="Training method (default: two_stage_freeze)")
    t.add_argument("--dataset", default=str(root / "data/dataset_v7_clean"))
    t.add_argument("--out", default=None, help="Output path (default: data/models/...)")
    t.add_argument("--weights", default=None)
    t.add_argument("--epochs", type=int, default=cfg["training"]["epochs"])
    t.add_argument("--batch", type=int, default=cfg["training"]["batch"])
    t.add_argument("--nice", action="store_true")
    t.add_argument("--unet", action="store_true")
    t.add_argument("--lr", type=float, default=cfg["training"]["learning_rate"])
    t.add_argument("--action-fusion", default="tc_roi", choices=["tc_roi", "gru_only", "gru_attended"])
    t.add_argument("--tth-scale", type=float, default=500.0)
    t.add_argument("--pos-weight", type=float, default=2.0)
    t.add_argument("--weighting", default="decoupled", choices=["kendall", "fixed", "hybrid", "decoupled"])
    t.add_argument("--stage1-epochs", type=int, default=None, help="Epochs for Stage 1 Aim in two_stage_freeze")
    t.add_argument("--stage2-epochs", type=int, default=None, help="Epochs for Stage 2 Action in two_stage_freeze")
    t.add_argument("--legacy", action="store_true", default=False, help="Fall back to legacy slow training loop")
    q = sub.add_parser("bench")
    q.add_argument("--weights")
    q.add_argument("--dataset", default=str(root / "data/dataset_v7"))
    q.add_argument("--iterations", type=int, default=500)
    q.add_argument("--suite", default="all", choices=["all", "latency", "mechanics"])
    q.add_argument("--temperature", type=float, default=1.0)
    e = sub.add_parser("evaluate")
    e.add_argument("--dataset", default=str(root / "data/dataset_v7"))
    e.add_argument("--weights", default=str(root / "data/models/tiny_policy_v6.pt"))
    v = sub.add_parser("preview")
    v.add_argument("--session", required=True)
    v.add_argument("--weights", default=str(root / "data/models/tiny_policy_v6.pt"))
    v.add_argument("--out", default=str(root / "data/previews/policy_preview_v7.mp4"))
    sub.add_parser("gui", help="open the recording-only GUI")
    l = sub.add_parser("label")
    l.add_argument("--session", required=True)
    r = sub.add_parser("rl")
    r.add_argument("--episodes", type=int, default=2500)
    a = p.parse_args()
    if a.cmd == "gui":
        run_gui()
    elif a.cmd == "label":
        label_session(a.session)
    elif a.cmd == "rl":
        run_rl(a.episodes)
    elif a.cmd == "build":
        build_dataset(a.source, a.out, cfg["input"]["width"], cfg["input"]["height"])
    elif a.cmd == "train":
        h = 160 if a.unet else cfg["model"]["hidden"]
        sc = 80 if a.unet else cfg["model"]["spatial_channels"]
        target_name = a.name or a.version
        if a.name or (target_name and target_name.startswith("Semga")):
            out_path = a.out or str(root / f"data/models/{target_name}.pt")
        else:
            out_path = a.out or str(root / f"data/models/unet_policy_{target_name}.pt")
        if getattr(a, "legacy", False):
            from osu2.train import train as legacy_train
            train_fn = legacy_train
        else:
            from osu2.fast_train import train as fast_train
            train_fn = fast_train
        train_fn(a.dataset, out_path, a.epochs, a.batch, cfg["training"]["sequence"], cfg["training"]["stride"], a.lr, a.nice, h, sc, use_unet=a.unet, pretrained=a.weights, action_fusion=a.action_fusion, tth_scale_ms=a.tth_scale, pos_weight=a.pos_weight, weighting=a.weighting, version=target_name, method=a.method, stage1_epochs=a.stage1_epochs, stage2_epochs=a.stage2_epochs)
    elif a.cmd=="evaluate": evaluate(a.dataset,a.weights,cfg["training"]["sequence"])
    elif a.cmd=="preview": preview(a.session,a.weights,a.out,cfg["training"]["sequence"])
    else: benchmark(a.weights, a.dataset, a.iterations, a.suite, a.temperature)
if __name__=="__main__": main()
