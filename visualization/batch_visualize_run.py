"""
Batch band / DOS visualization for a DeePTB training run.

Usage
-----
Band plots for all train + validation structures::

    python visualization/batch_visualize_run.py \
        --run-dir RUN \
        --backend band \
        --mode full

Band + DOS plots::

    python visualization/batch_visualize_run.py \
        --run-dir RUN \
        --backend both \
        --mode full

One validation case for testing::

    python visualization/batch_visualize_run.py \
        --run-dir RUN \
        --backend band \
        --mode full \
        --splits val \
        --limit 1

The run input JSON and checkpoint are auto-detected unless supplied explicitly.
The plotting backends are imported once and the DeePTB checkpoint is loaded once,
then reused for every structure in the same Python process.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

SET_RE = re.compile(r"^set\.(\d+)$")
EP_RE = re.compile(r"\.ep(\d+)\.pth$")


def natural_set_key(path: Path) -> int:
    m = SET_RE.match(path.name)
    return int(m.group(1)) if m else 10**12


def sanitize_label(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return text.strip("_") or "case"


def defect_label(set_dir: Path) -> str:
    report = set_dir / "conversion_report.json"
    suffix = None
    if report.is_file():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            source = data.get("source_case")
            if source:
                suffix = Path(source).name
        except Exception:
            suffix = None
    set_token = set_dir.name.replace(".", "")
    return f"{set_token}_{sanitize_label(suffix)}" if suffix else set_token


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def resolve_input_json(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    preferred = [
        run_dir / "input.json",
        run_dir / "input_train.json",
        run_dir / "train.json",
    ]
    for path in preferred:
        if path.is_file():
            return path.resolve()

    candidates = sorted(
        p for p in run_dir.glob("*.json")
        if p.is_file() and "manifest" not in p.name.lower()
    )
    input_like = [p for p in candidates if p.name.lower().startswith("input")]
    if len(input_like) == 1:
        return input_like[0].resolve()
    if len(candidates) == 1:
        return candidates[0].resolve()

    names = ", ".join(p.name for p in candidates) or "none"
    raise FileNotFoundError(
        f"Could not uniquely identify the DeePTB input JSON in {run_dir}. "
        f"Candidates: {names}. Use --input-json."
    )


def checkpoint_rank(path: Path) -> tuple[int, int, float]:
    name = path.name
    if name == "nnsk.best.pth":
        return (4, 0, path.stat().st_mtime)
    if name == "nnsk.latest.pth":
        return (3, 0, path.stat().st_mtime)
    m = EP_RE.search(name)
    if m:
        return (2, int(m.group(1)), path.stat().st_mtime)
    return (1, 0, path.stat().st_mtime)


def resolve_checkpoint(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    search_dirs = [
        run_dir / "output" / "checkpoint",
        run_dir / "checkpoint",
        run_dir / "output",
    ]
    candidates: list[Path] = []
    for directory in search_dirs:
        if directory.is_dir():
            candidates.extend(directory.glob("*.pth"))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found below {run_dir}. Use --checkpoint."
        )

    candidates = sorted(
        {p.resolve() for p in candidates},
        key=checkpoint_rank,
        reverse=True,
    )
    return candidates[0]


def resolve_dataset_root(raw: str | Path, input_json: Path, run_dir: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {resolved}")
        return resolved

    candidates = [
        (input_json.parent / path).resolve(),
        (run_dir / path).resolve(),
        path.resolve(),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve dataset root {raw!r}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def dataset_roots(config: dict[str, Any], input_json: Path, run_dir: Path) -> dict[str, Path]:
    data_options = config.get("data_options")
    if not isinstance(data_options, dict):
        raise KeyError("Input JSON has no data_options object")

    roots: dict[str, Path] = {}
    key_map = {
        "train": ("train",),
        "val": ("validation", "val"),
    }
    for out_name, keys in key_map.items():
        section = None
        for key in keys:
            value = data_options.get(key)
            if isinstance(value, dict):
                section = value
                break
        if section is None:
            continue
        root = section.get("root")
        if root is None:
            continue
        roots[out_name] = resolve_dataset_root(root, input_json, run_dir)

    if "train" not in roots and "val" not in roots:
        raise KeyError(
            "Could not find data_options.train.root or "
            "data_options.validation.root in the input JSON"
        )
    return roots


def discover_sets(root: Path) -> list[Path]:
    sets = [
        p for p in root.iterdir()
        if p.is_dir() and SET_RE.match(p.name)
    ]
    sets.sort(key=natural_set_key)
    return sets


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_main(module, argv: list[str]) -> None:
    old_argv = sys.argv
    try:
        sys.argv = [str(Path(module.__file__).name), *argv]
        module.main()
    finally:
        sys.argv = old_argv
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass


def copy_flat_pngs(case_output: Path, flat_dir: Path, prefix: str) -> list[str]:
    copied = []
    flat_dir.mkdir(parents=True, exist_ok=True)
    for png in sorted(case_output.glob("*.png")):
        dst = flat_dir / f"{prefix}_{png.name}"
        shutil.copy2(png, dst)
        copied.append(str(dst))
    return copied


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch band / DOS visualization for one DeePTB training run."
    )
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--input-json", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--backend",
        choices=("band", "dos", "both"),
        default="band",
    )
    p.add_argument(
        "--mode",
        choices=("supervised", "full"),
        default="full",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val"),
        default=("train", "val"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: RUN/visualization",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of structures per selected split.",
    )
    p.add_argument("--dpi", type=int, default=300)

    p.add_argument(
        "--plot-content",
        choices=("comparison", "prediction", "ground-truth"),
        default="comparison",
        help="Used by the DOS backend.",
    )
    p.add_argument(
        "--dos-mode",
        choices=("total", "all"),
        default="all",
    )
    p.add_argument(
        "--kmesh",
        nargs=3,
        type=int,
        default=(30, 30, 1),
        metavar=("NX", "NY", "NZ"),
    )
    p.add_argument("--sigma", type=float, default=0.10)

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve run/checkpoint/datasets and print the case plan only.",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if one structure/backend fails.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.dpi < 50:
        raise ValueError("--dpi must be >= 50")
    if args.sigma <= 0:
        raise ValueError("--sigma must be > 0")
    if any(x < 1 for x in args.kmesh):
        raise ValueError("--kmesh values must be >= 1")

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    input_json = resolve_input_json(run_dir, args.input_json)
    config = load_json(input_json)
    roots = dataset_roots(config, input_json, run_dir)
    checkpoint = resolve_checkpoint(run_dir, args.checkpoint)

    selected_splits = list(dict.fromkeys(args.splits))
    cases: list[tuple[str, Path]] = []
    for split in selected_splits:
        if split not in roots:
            raise KeyError(f"Requested split {split!r} is not defined in {input_json}")
        split_sets = discover_sets(roots[split])
        if args.limit is not None:
            split_sets = split_sets[: args.limit]
        cases.extend((split, p) for p in split_sets)

    output_root = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_dir / "visualization"
    )

    backends = ["band", "dos"] if args.backend == "both" else [args.backend]

    print(f"Run directory: {run_dir}")
    print(f"Input JSON:    {input_json}")
    print(f"Checkpoint:    {checkpoint}")
    for split in selected_splits:
        n = sum(1 for s, _ in cases if s == split)
        print(f"{split:>10}:    {roots[split]}  ({n} set(s))")
    print(f"Backend:       {args.backend}")
    print(f"Mode:          {args.mode}")
    print(f"Output:        {output_root}")

    if args.dry_run:
        print("\nPlanned cases:")
        for split, set_dir in cases:
            print(f"  {split:5s}  {set_dir.name}  {defect_label(set_dir)}")
        print("\nDry run only.")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    flat_dir = output_root / "flat_png"

    script_dir = Path(__file__).resolve().parent
    band_path = script_dir / "band_plot.py"
    dos_path = script_dir / "band_dos_compare.py"

    band_module = None
    dos_module = None
    if "band" in backends:
        if not band_path.is_file():
            raise FileNotFoundError(band_path)
        band_module = load_module(band_path, "_deeptb4hbn_band_plot")
    if "dos" in backends:
        if not dos_path.is_file():
            raise FileNotFoundError(dos_path)
        dos_module = load_module(dos_path, "_deeptb4hbn_band_dos_compare")

    from dptb.nn.build import build_model
    print(f"\nLoading DeePTB checkpoint once: {checkpoint}")
    model = build_model(checkpoint=str(checkpoint))
    print(f"Model device: {model.device}")

    def cached_build_model(*_args, **_kwargs):
        return model

    if band_module is not None:
        band_module.build_model = cached_build_model
    if dos_module is not None:
        dos_module.build_model = cached_build_model

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    total_jobs = len(cases) * len(backends)
    job_index = 0
    t_all = time.perf_counter()

    for split, set_dir in cases:
        label = defect_label(set_dir)

        for backend in backends:
            job_index += 1
            case_output = output_root / backend / split / label
            case_output.mkdir(parents=True, exist_ok=True)

            print(
                f"\n[{job_index}/{total_jobs}] "
                f"{split}/{set_dir.name} -> {backend}"
            )
            t0 = time.perf_counter()
            status = "ok"
            error = ""

            try:
                if backend == "band":
                    argv = [
                        "--checkpoint", str(checkpoint),
                        "--set", str(set_dir),
                        "--output", str(case_output),
                        "--mode", args.mode,
                        "--dpi", str(args.dpi),
                    ]
                    run_main(band_module, argv)
                else:
                    argv = [
                        "--checkpoint", str(checkpoint),
                        "--set", str(set_dir),
                        "--output", str(case_output),
                        "--plot-content", args.plot_content,
                        "--mode", args.mode,
                        "--dos-mode", args.dos_mode,
                        "--kmesh", *(str(x) for x in args.kmesh),
                        "--sigma", str(args.sigma),
                        "--dpi", str(args.dpi),
                    ]
                    run_main(dos_module, argv)

                copied = copy_flat_pngs(
                    case_output,
                    flat_dir,
                    f"{split}_{label}_{backend}",
                )
                if not copied:
                    raise RuntimeError(
                        f"{backend} backend completed but produced no PNG in {case_output}"
                    )

            except Exception as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                print(f"FAILED: {error}")
                failures.append(
                    {
                        "split": split,
                        "set": set_dir.name,
                        "label": label,
                        "backend": backend,
                        "error": error,
                        "output": str(case_output),
                    }
                )
                if args.fail_fast:
                    raise

            elapsed = time.perf_counter() - t0
            records.append(
                {
                    "split": split,
                    "set": set_dir.name,
                    "label": label,
                    "backend": backend,
                    "mode": args.mode,
                    "status": status,
                    "elapsed_s": f"{elapsed:.3f}",
                    "output": str(case_output),
                    "error": error,
                }
            )
            print(f"{status.upper()} in {elapsed:.1f} s")

    summary_fields = [
        "split", "set", "label", "backend", "mode",
        "status", "elapsed_s", "output", "error",
    ]
    write_csv(
        output_root / "batch_visualization_summary.csv",
        records,
        summary_fields,
    )
    write_csv(
        output_root / "batch_visualization_failures.csv",
        failures,
        ["split", "set", "label", "backend", "error", "output"],
    )

    manifest = {
        "run_dir": str(run_dir),
        "input_json": str(input_json),
        "checkpoint": str(checkpoint),
        "dataset_roots": {k: str(v) for k, v in roots.items()},
        "backend": args.backend,
        "mode": args.mode,
        "splits": selected_splits,
        "limit_per_split": args.limit,
        "dos": {
            "plot_content": args.plot_content,
            "dos_mode": args.dos_mode,
            "kmesh": list(args.kmesh),
            "sigma_eV": args.sigma,
        },
        "n_jobs": total_jobs,
        "n_failed": len(failures),
    }
    (output_root / "batch_visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    elapsed_all = time.perf_counter() - t_all
    print("\n" + "=" * 72)
    print("Batch visualization complete")
    print("=" * 72)
    print(f"Jobs:      {total_jobs}")
    print(f"Failures:  {len(failures)}")
    print(f"Elapsed:   {elapsed_all:.1f} s")
    print(f"Output:    {output_root}")


if __name__ == "__main__":
    main()
