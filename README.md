# DeePTB4hBN

Utilities developed for the hBN defect-family DeePTB workflow, linking FHI-aims band-structure calculations to DeePTB dataset preparation, model evaluation, and band-edge diagnostics.

## Repository status

This repository currently consolidates the **PC demonstration-stage data conversion and evaluation pipeline**. The scripts here are sufficient to:

1. convert individual or batched FHI-aims calculations into one-frame DeePTB datasets;
2. compare DeePTB predictions with the supervised DFT targets;
3. inspect the full available non-core spectrum for selected structures;
4. compare checkpoints band-by-band; and
5. compare DFT-defined host-like VBM, CBM, and band-gap predictions between two model sources.

They are **not yet sufficient by themselves to launch the scaled NSCC training stage**. The next repository update should add the canonical DeePTB training input/configuration used for the final PC baseline and the NSCC PBS submission script. Keeping those files separate from the analysis utilities will make the transition from the 50-epoch demo to longer, reproducible training runs explicit.

## Workflow

```text
FHI-aims calculations
        |
        v
aims_to_deeptb.py
        |
        +-- batch_aims_to_deeptb.py   (many calculations)
        |
        v
DeePTB set.* datasets
        |
        v
DeePTB training
        |
        +-- evaluate_model.py         (batch supervised-target metrics)
        +-- band_plot.py              (single-case DFT/model band overlay)
        +-- evaluate_band_error.py    (single-case full-spectrum band errors)
        +-- compare_frontiers.py      (VBM/CBM/gap comparison)
```

## Scripts

| Script | Purpose | Typical scope |
| --- | --- | --- |
| `scripts/aims_to_deeptb.py` | Convert one FHI-aims calculation to DeePTB-SK format with explicit non-core band-selection policies | One calculation |
| `scripts/batch_aims_to_deeptb.py` | Recursively convert many calculations into `set.XXXXXX` directories and write manifests | Dataset preparation |
| `scripts/band_plot.py` | Overlay FHI-aims and DeePTB bands on the same k-path in supervised or full non-core mode | One structure/checkpoint |
| `scripts/evaluate_model.py` | Batch evaluation of a checkpoint against stored train/validation targets, including split-, quartile-, and band-resolved metrics | Many structures |
| `scripts/evaluate_band_error.py` | Band-index-resolved error analysis, including unsupervised non-core bands and optional two-checkpoint comparison | One structure, one/two checkpoints |
| `scripts/compare_frontiers.py` | Compare two prediction sources using DFT-defined host-like VBM/CBM/gap targets | Many structures, two sources |

### `evaluate_model.py` vs `evaluate_band_error.py`

These scripts are complementary rather than duplicates.

- `evaluate_model.py` is the **batch evaluator**. It scans train/validation `set.*` directories, evaluates one checkpoint, and aggregates errors across structures. Its DFT reference is the stored converted `eigenvalues.npy`, so it evaluates the supervised target window.
- `evaluate_band_error.py` is the **single-structure spectral diagnostic**. In `--mode full`, it reloads the original FHI-aims bands, strips inferred core bands, and evaluates non-core bands above the training cutoff. It can also compare two checkpoints directly.

For scaled training, keep both: use `evaluate_model.py` for routine epoch/model selection and `evaluate_band_error.py` for detailed extrapolation checks on representative defects.

## Quick start

### 1. Convert one FHI-aims calculation

Recommended adaptive non-core window:

```bash
python scripts/aims_to_deeptb.py CASE_DIR -o OUTPUT_DIR \
    --band-policy adaptive-factor --band-factor 2.0
```

Alternative policies:

```bash
# Keep all available non-core bands
python scripts/aims_to_deeptb.py CASE_DIR -o OUTPUT_DIR \
    --band-policy all-noncore

# Keep exactly N non-core bands
python scripts/aims_to_deeptb.py CASE_DIR -o OUTPUT_DIR \
    --band-policy fixed-noncore --noncore-bands N
```

The converter writes:

```text
xdat.traj
kpoints.npy
eigenvalues.npy
info.json
conversion_report.json
```

### 2. Convert a calculation tree

```bash
python scripts/batch_aims_to_deeptb.py INPUT_ROOT OUTPUT_ROOT \
    --converter scripts/aims_to_deeptb.py \
    --band-policy adaptive-factor --band-factor 2.0
```

Resume an interrupted conversion with the same selection policy:

```bash
python scripts/batch_aims_to_deeptb.py INPUT_ROOT OUTPUT_ROOT \
    --converter scripts/aims_to_deeptb.py \
    --band-policy adaptive-factor --band-factor 2.0 \
    --resume
```

Each calculation is stored as one `set.XXXXXX` directory. `manifest.json` and `manifest.csv` record source paths, conversion status, composition, band-selection metadata, and failures.

### 3. Batch-evaluate a checkpoint

```bash
python scripts/evaluate_model.py \
    --checkpoint RUN/checkpoint/nnsk.epN.pth \
    --train DATA/train \
    --val DATA/val \
    --output RESULTS/eval_epN \
    --make-plots
```

This is the main evaluator for convergence studies and train/validation comparisons during longer training.

### 4. Plot a representative band structure

Supervised target only:

```bash
python scripts/band_plot.py \
    --checkpoint RUN/checkpoint/nnsk.epN.pth \
    --set DATA/set.000000 \
    --output RESULTS/band_plot \
    --mode supervised
```

Full available non-core spectrum:

```bash
python scripts/band_plot.py \
    --checkpoint RUN/checkpoint/nnsk.epN.pth \
    --set DATA/set.000000 \
    --output RESULTS/band_plot_full \
    --mode full
```

`--mode full` uses `source_case` in `conversion_report.json` unless `--aims-dir` is supplied.

### 5. Inspect error versus band index

```bash
python scripts/evaluate_band_error.py \
    --checkpoint RUN/checkpoint/nnsk.epN.pth \
    --set DATA/set.000000 \
    --output RESULTS/band_error_epN \
    --mode full
```

Compare two checkpoints on exactly the same structure:

```bash
python scripts/evaluate_band_error.py \
    --checkpoint RUN_A/checkpoint/nnsk.epN.pth \
    --checkpoint2 RUN_B/checkpoint/nnsk.epM.pth \
    --set DATA/set.000000 \
    --output RESULTS/band_error_compare \
    --mode full
```

### 6. Compare host-like band edges between two model sources

```bash
python scripts/compare_frontiers.py \
    --source-a RUN_A \
    --source-b RUN_B \
    --train DATA/train \
    --val DATA/val \
    --output RESULTS/frontier_compare \
    --make-plots
```

Each source can be a checkpoint, a training-output directory containing `nnsk.ep*.pth`, or an `evaluate_model.py` output directory with reusable `_deeptb_work` predictions.

## Assumptions and cautions

The present workflow is specialized for the hBN defect dataset rather than being a general FHI-aims/DeePTB converter.

- Core-band inference is defined for H/B/C/N/O: H contributes zero inferred frozen 1s spatial core bands; B/C/N/O contribute one each.
- The batch converter skips cases containing `band2*.out` by default. The evaluation scripts likewise avoid silently assuming how a second spin channel should be treated.
- `band_plot.py` uses the hBN `M -> Gamma -> K -> M` high-symmetry path convention.
- The converter preserves the original FHI-aims energy gauge. Evaluation scripts apply explicit energy-offset alignments where required by the DeePTB eigenvalue-loss convention.
- `compare_frontiers.py` uses a **spectral heuristic** to identify host-like VBM/CBM bands in defect systems. Its default host-gap plausibility reference is 4.672 eV with a +/-0.5 eV tolerance; override these values when a different reference is appropriate. Localization-sensitive quantities such as IPR or projected character remain preferable for physically ambiguous defect states.

## Python dependencies

The scripts use:

- Python 3
- NumPy
- ASE
- PyTorch
- Matplotlib (plotting/evaluation scripts)
- DeePTB and its runtime dependencies

Use the same DeePTB environment for training and evaluation to avoid checkpoint/API incompatibilities.

## Next step: NSCC scaling

Before starting the larger NSCC campaign, add the final PC-baseline training configuration and an NSCC launch layer, for example:

```text
input/
    hbn_nnsk_baseline.json

nscc/
    train_hbn_a100.pbs
```

The NSCC job should preserve the dataset split and model definition used by the PC baseline, while making epochs, checkpoint frequency, resume behavior, run directory, and resource request explicit. That will let the first cluster run answer a clean question: **does the 50-epoch trend continue toward convergence when the same model is trained substantially longer?**
