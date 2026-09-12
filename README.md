# DeePTB4hBN

Utilities developed for the hBN defect-family DeePTB workflow, linking FHI-aims band-structure calculations to DeePTB dataset preparation, model evaluation, and electronic-structure diagnostics.

## Repository status

This repository currently consolidates the **PC demonstration-stage data conversion and evaluation pipeline**. The scripts here are sufficient to:

1. convert individual or batched FHI-aims calculations into one-frame DeePTB datasets;
2. compare DeePTB predictions with supervised or full available non-core DFT bands;
3. compare FHI-aims and DeePTB band structures together with total/species-projected DOS;
4. compare checkpoints band-by-band; and
5. compare DFT-defined host-like VBM, CBM, and band-gap predictions between model sources.

They are **not yet sufficient by themselves to launch the scaled NSCC training stage**. The planned NSCC update should add the canonical training configuration and PBS submission scripts used for longer, reproducible runs.

## Workflow

```text
FHI-aims calculations
        |
        v
aims_to_deeptb.py
        |
        +-- batch_aims_to_deeptb.py
        |
        v
DeePTB set.* datasets
        |
        v
DeePTB training
        |
        +-- evaluate_model.py
        +-- band_plot.py
        +-- band_dos_compare.py
        +-- evaluate_band_error.py
        +-- compare_frontiers.py
```

## Scripts

| Script | Purpose | Typical scope |
| --- | --- | --- |
| `conversion/aims_to_deeptb.py` | Convert one FHI-aims calculation to DeePTB-SK format with selectable non-core band policies | One calculation |
| `conversion/batch_aims_to_deeptb.py` | Recursively convert many calculations into `set.XXXXXX` directories and write manifests | Dataset preparation |
| `visualization/band_plot.py` | Overlay FHI-aims and DeePTB bands in supervised or full non-core mode | One structure/checkpoint |
| `visualization/band_dos_compare.py` | Plot FHI-aims ground truth, DeePTB prediction, or their comparison with adaptive total/species DOS panels | One structure/checkpoint |
| `evaluation/evaluate_model.py` | Batch checkpoint evaluation against stored train/validation targets | Many structures |
| `evaluation/evaluate_band_error.py` | Band-index-resolved error analysis, including unsupervised non-core bands and optional two-checkpoint comparison | One structure, one/two checkpoints |
| `evaluation/compare_frontiers.py` | Compare model sources using DFT-defined host-like VBM/CBM/gap targets | Many structures, two sources |

### `evaluate_model.py` vs `evaluate_band_error.py`

These scripts are complementary:

- `evaluate_model.py` is the **batch evaluator**. It scans train/validation `set.*` directories and evaluates one checkpoint against the stored converted targets.
- `evaluate_band_error.py` is the **single-structure spectral diagnostic**. In `--mode full`, it reloads the original FHI-aims bands and evaluates available non-core bands above the training cutoff. It can also compare two checkpoints directly.

For longer training, use `evaluate_model.py` for routine model selection and `evaluate_band_error.py` for detailed extrapolation checks on representative defects.

## Quick start

### 1. Convert one FHI-aims calculation

Recommended adaptive non-core window:

```bash
python conversion/aims_to_deeptb.py CASE_DIR -o OUTPUT_DIR \
    --band-policy adaptive-factor --band-factor 2.0
```

Alternative policies:

```bash
# Keep all available non-core bands
python conversion/aims_to_deeptb.py CASE_DIR -o OUTPUT_DIR \
    --band-policy all-noncore

# Keep exactly N non-core bands
python conversion/aims_to_deeptb.py CASE_DIR -o OUTPUT_DIR \
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
python conversion/batch_aims_to_deeptb.py INPUT_ROOT OUTPUT_ROOT \
    --band-policy adaptive-factor --band-factor 2.0
```

Resume with the same selection policy:

```bash
python conversion/batch_aims_to_deeptb.py INPUT_ROOT OUTPUT_ROOT \
    --band-policy adaptive-factor --band-factor 2.0 \
    --resume
```

Each calculation is stored as one `set.XXXXXX` directory. `manifest.json` and `manifest.csv` record source paths, conversion status, composition, band-selection metadata, and failures.

### 3. Batch-evaluate a checkpoint

```bash
python evaluation/evaluate_model.py \
    --checkpoint RUN/checkpoint/nnsk.epN.pth \
    --train DATA/train \
    --val DATA/val \
    --output RESULTS/eval_epN \
    --make-plots
```

### 4. Plot a representative band structure

Supervised target only:

```bash
python visualization/band_plot.py \
    --checkpoint RUN/checkpoint/nnsk.epN.pth \
    --set DATA/set.000000 \
    --output RESULTS/band_plot \
    --mode supervised
```

Full available non-core spectrum:

```bash
python visualization/band_plot.py \
    --checkpoint RUN/checkpoint/nnsk.epN.pth \
    --set DATA/set.000000 \
    --output RESULTS/band_plot_full \
    --mode full
```

`--mode full` uses `source_case` in `conversion_report.json` unless `--aims-dir` is supplied.

### 5. Compare band structure and DOS

Ground truth only:

```bash
python visualization/band_dos_compare.py \
    --set DATA/set.000000 \
    --output RESULTS/band_dos_dft \
    --plot-content ground-truth \
    --mode full \
    --dos-mode all
```

DeePTB prediction only:

```bash
python visualization/band_dos_compare.py \
    --checkpoint RUN/checkpoint/nnsk.epN.pth \
    --set DATA/set.000000 \
    --output RESULTS/band_dos_pred \
    --plot-content prediction \
    --mode full \
    --dos-mode all \
    --kmesh 30 30 1 \
    --sigma 0.10
```

FHI-aims vs DeePTB:

```bash
python visualization/band_dos_compare.py \
    --checkpoint RUN/checkpoint/nnsk.epN.pth \
    --set DATA/set.000000 \
    --output RESULTS/band_dos_compare \
    --plot-content comparison \
    --mode full \
    --dos-mode all \
    --kmesh 30 30 1 \
    --sigma 0.10
```

`--dos-mode total` plots only total DOS; `--dos-mode all` adds one species-projected panel for every species present in the structure.

The script prefers the ordinary FHI-aims `KS_DOS_total.dat` and `<species>_l_proj_dos.dat` files so the DOS and FHI-aims bands share the same energy reference. If the DOS calculation covers a narrower energy window than the plotted bands, the script prints a warning rather than treating missing DOS as an absence of states.

### 6. Inspect error versus band index

```bash
python evaluation/evaluate_band_error.py \
    --checkpoint RUN/checkpoint/nnsk.epN.pth \
    --set DATA/set.000000 \
    --output RESULTS/band_error_epN \
    --mode full
```

Compare two checkpoints:

```bash
python evaluation/evaluate_band_error.py \
    --checkpoint RUN_A/checkpoint/nnsk.epN.pth \
    --checkpoint2 RUN_B/checkpoint/nnsk.epM.pth \
    --set DATA/set.000000 \
    --output RESULTS/band_error_compare \
    --mode full
```

### 7. Compare host-like band edges

```bash
python evaluation/compare_frontiers.py \
    --source-a RUN_A \
    --source-b RUN_B \
    --train DATA/train \
    --val DATA/val \
    --output RESULTS/frontier_compare \
    --make-plots
```

Each source can be a checkpoint, a training-output directory containing `nnsk.ep*.pth`, or an `evaluate_model.py` output directory with reusable `_deeptb_work` predictions.

## Assumptions and cautions

The workflow is specialized for the hBN defect dataset rather than being a general FHI-aims/DeePTB interface.

- Core-band inference is defined for H/B/C/N/O: H contributes zero inferred frozen 1s spatial core bands; B/C/N/O contribute one each.
- The converters/evaluators avoid silently combining `band2*.out` with the current single-channel workflow.
- The visualization scripts use the hBN `M -> Gamma -> K -> M` high-symmetry path convention.
- `band_dos_compare.py` keeps FHI-aims in its native band/DOS energy gauge and rigidly aligns DeePTB to it when `--align loss` is used.
- FHI-aims DOS files cover only the energy window requested in the underlying DFT calculation; the script warns when this does not span the plotted band range.
- `compare_frontiers.py` uses a spectral heuristic to identify host-like VBM/CBM bands in defect systems. Localization-sensitive quantities such as IPR or projected character remain preferable for ambiguous defect states.

## Python dependencies

- Python 3
- NumPy
- ASE
- PyTorch
- Matplotlib
- DeePTB and its runtime dependencies

Use the same DeePTB environment for training and evaluation to avoid checkpoint/API incompatibilities.

## Next step: NSCC scaling

Before the larger NSCC campaign, add the final PC-baseline training configuration and NSCC launch layer, for example:

```text
input/
    hbn_nnsk_baseline.json

nscc/
    train_hbn_a100.pbs
```

The cluster runs should preserve the selected dataset split/model definition while making epochs, checkpoint frequency, resume behavior, output path, and resource request explicit.
