# DeePTB4hBN

Utilities developed for the hBN defect-family DeePTB workflow, linking FHI-aims band-structure calculations to DeePTB dataset preparation, optional energy-region weighting, model evaluation, and electronic-structure diagnostics.

## Workflow

```text
FHI-aims calculations
        |
        v
conversion/aims_to_deeptb.py
        |
        +-- conversion/batch_aims_to_deeptb.py
        |
        v
DeePTB set.* datasets
        |
        +-- preprocessing/split_train_val.py
        |       |
        |       +-- train/set.*
        |       +-- val/set.*
        |
        +-- preprocessing/weight_band_regions.py      optional
        |       |
        |       +-- info.unweighted.json
        |       +-- info.weighted.json
        |       +-- band_regions.json
        |       v
        |   visualization/visualize_band_regions.py
        |
        v
DeePTB training
        |
        +-- evaluation/evaluate_model.py
        +-- evaluation/evaluate_band_error.py
        +-- evaluation/compare_frontiers.py
        +-- visualization/band_plot.py
        +-- visualization/band_dos_compare.py
```

The converted eigenvalue targets remain unchanged when band-region weighting is enabled. Weighting is introduced only through DeePTB's existing `bandinfo.emin` / `bandinfo.emax` metadata and the training-time `eout_weight` setting.

## Scripts

| Script | Purpose | Typical scope |
| --- | --- | --- |
| `conversion/aims_to_deeptb.py` | Convert one FHI-aims calculation to DeePTB-SK format with selectable non-core band policies | One calculation |
| `conversion/batch_aims_to_deeptb.py` | Recursively convert many calculations into `set.XXXXXX` directories and write manifests | Dataset preparation |
| `preprocessing/split_train_val.py` | Create reproducible train/validation trees by explicit set list or seeded random split, with copy or symlink materialization | Dataset preparation |
| `preprocessing/weight_band_regions.py` | Detect the lower-spectrum / upper-spectrum separation, generate reversible weighted metadata, and switch between weighted and unweighted `info.json` states | One set or a dataset tree |
| `visualization/visualize_band_regions.py` | Batch QA of detected weighting regions using selected converted bands or the full available non-core FHI-aims spectrum | One set or a dataset tree |
| `visualization/band_plot.py` | Overlay FHI-aims and DeePTB bands in supervised or full non-core mode | One structure/checkpoint |
| `visualization/band_dos_compare.py` | Plot FHI-aims ground truth, DeePTB prediction, or their comparison with total/species-projected DOS | One structure/checkpoint |
| `evaluation/evaluate_model.py` | Batch checkpoint evaluation against stored train/validation targets | Many structures |
| `evaluation/evaluate_band_error.py` | Band-index-resolved error analysis, including unsupervised non-core bands and optional two-checkpoint comparison | One structure, one/two checkpoints |
| `evaluation/compare_frontiers.py` | Compare model sources using DFT-defined host-like VBM/CBM/gap targets | Many structures, two sources |

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

### 3. Split converted data into train/validation sets

Create a reproducible 80/20 split:

```bash
python preprocessing/split_train_val.py CONVERTED_DATA SPLIT_DATA \
    --val-fraction 0.2 --seed 42
```

For a fixed validation set, specify the set IDs directly:

```bash
python preprocessing/split_train_val.py CONVERTED_DATA SPLIT_DATA \
    --val-sets \
    set.000007 set.000013 set.000017 set.000023 set.000027 \
    set.000030 set.000037 set.000040 set.000044 set.000048
```

A text file can also be used:

```bash
python preprocessing/split_train_val.py CONVERTED_DATA SPLIT_DATA \
    --val-list validation_sets.txt
```

Inspect a proposed split without writing files:

```bash
python preprocessing/split_train_val.py CONVERTED_DATA SPLIT_DATA \
    --val-fraction 0.2 --seed 42 --dry-run
```

The default materialization mode is `copy`, which is the most portable choice for NSCC. To avoid duplicating data when the source and split trees will remain together:

```bash
python preprocessing/split_train_val.py CONVERTED_DATA SPLIT_DATA \
    --val-fraction 0.2 --seed 42 --mode symlink
```

The source dataset is never modified. The output layout is:

```text
SPLIT_DATA/
    train/
        set.XXXXXX/
    val/
        set.XXXXXX/
    split_manifest.json
    split_manifest.csv
```

The manifest records the exact train/validation membership, selection method, random seed when applicable, and materialization mode. Existing `train/` or `val/` directories are not replaced unless `--overwrite` is supplied.

Use a sibling output directory by default. Nested output is blocked unless `--allow-nested-output` is explicitly supplied, because recursive preprocessing tools could otherwise encounter duplicate `set.*` directories.

## Optional energy-region weighting

The current hBN workflow can down-weight the deep lower-spectrum manifold while retaining the complete converted target and the same `2s + 2p + d*` DeePTB basis.

The detector first searches for a sufficiently large internal empty gap. If none is found, it falls back to a low-density corridor identified from the retained band spectrum. The selected split is converted to DeePTB's minimum-aligned energy gauge and written as `bandinfo.emin`; `bandinfo.emax` is placed above the highest selected target energy.

### Inspect without modifying files

```bash
python preprocessing/weight_band_regions.py DATA --recursive
```

### Generate and activate weighted metadata

```bash
python preprocessing/weight_band_regions.py DATA --recursive --apply
```

For each `set.XXXXXX`, the script keeps:

```text
info.json                 # active metadata read by DeePTB
info.unweighted.json      # preserved equal-weight baseline
info.weighted.json        # generated weighted configuration
band_regions.json         # detector parameters, split information, checks, hashes
```

Only the DeePTB-required energy-window fields are added to `info.weighted.json`. Detector diagnostics and provenance are stored separately in `band_regions.json`.

The default detector parameters are:

```text
min gap                  5.0 eV
minimum split offset     5.0 eV above E0
maximum search offset   15.0 eV above E0
grid step                0.05 eV
minimum sparse width     1.0 eV
sparse crossing limit    min(4, max(2, ceil(0.02 * Nbands)))
emax margin              1.0 eV
```

The script performs consistency checks before activation, including

```text
split_offset == split_energy - E0
emin         == split_offset
emax         >  highest aligned selected-band energy
emax         >  emin > 0
```

and verifies active/canonical metadata switches using SHA-256 hashes.

### Switch back to equal weighting

```bash
python preprocessing/weight_band_regions.py SPLIT_DATA --recursive --undo
```

Reactivate an already-generated weighted configuration without rerunning the detector:

```bash
python preprocessing/weight_band_regions.py SPLIT_DATA --recursive --activate-weighted
```

Inspect the active metadata state:

```bash
python preprocessing/weight_band_regions.py SPLIT_DATA --recursive --status
```

If the active `info.json` matches neither canonical copy, the script reports an `unknown` state and refuses to overwrite it unless `--force-switch` is explicitly supplied.

### Configure the DeePTB loss

The detector records a recommended weight for provenance, but `eout_weight` remains a training hyperparameter and must be set in the DeePTB input file:

```json
"loss_options": {
  "train": {
    "method": "eigvals",
    "eout_weight": 0.5
  }
}
```

With an energy window present, the current DeePTB eigenvalue loss behaves as

```text
loss = MSE(in-window states) + eout_weight * MSE(out-of-window states)
```

The eigenvalue arrays themselves are not rewritten or truncated.

## Band-region visual QA

Before using weighted metadata for training, inspect the detected separation in batches.

### Selected converted target

```bash
python visualization/visualize_band_regions.py DATA --mode selected
```

This plots exactly the retained `eigenvalues.npy` spectrum used by the detector.

### Full available non-core FHI-aims spectrum

```bash
python visualization/visualize_band_regions.py DATA --mode full
```

Full mode reloads `band1*.out`, strips the inferred 1s core bands, and verifies that the converted target is the expected prefix of the reconstructed non-core spectrum.

By default the visualizer reads the canonical `info.weighted.json`. To inspect the metadata currently active in DeePTB:

```bash
python visualization/visualize_band_regions.py DATA \
    --mode selected \
    --window-source active
```

The plots use:

- black lines for FHI-aims bands;
- light orange shading for the detected low-density candidate region;
- dashed red lines for `emin` and `emax`;
- per-panel annotations for the split, offset, region width, confidence, and active metadata state.

Use gray shading instead:

```bash
python visualization/visualize_band_regions.py DATA \
    --mode selected \
    --shade-color gray
```

Batch runs write individual PNGs, a CSV summary, and 4x4 contact sheets by default.

## Model evaluation

### Batch-evaluate a checkpoint

```bash
python evaluation/evaluate_model.py \
    --checkpoint RUN/checkpoint/nnsk.epN.pth \
    --train DATA/train \
    --val DATA/val \
    --output RESULTS/eval_epN \
    --make-plots
```

### Plot a representative band structure

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

### Compare band structure and DOS

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

The script prefers the ordinary FHI-aims `KS_DOS_total.dat` and `<species>_l_proj_dos.dat` files so the DOS and FHI-aims bands share the same energy reference. If the DOS calculation covers a narrower energy window than the plotted bands, the script warns rather than treating missing DOS as an absence of states.

### Inspect error versus band index

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

### Compare host-like band edges

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
- The converters and evaluators avoid silently combining `band2*.out` with the current single-channel workflow.
- The visualization scripts use the hBN `M -> Gamma -> K -> M` high-symmetry path convention.
- `weight_band_regions.py` detects a spectral separation, not a chemically exact 2s projector. The low-density region is a detection aid; the selected `emin` is the actual weighting boundary.
- Weighted and unweighted validation losses are not directly comparable when different `eout_weight` settings are used. Preserve an unweighted evaluation route when comparing training strategies.
- `band_dos_compare.py` keeps FHI-aims in its native band/DOS energy gauge and rigidly aligns DeePTB to it when `--align loss` is used.
- FHI-aims DOS files cover only the requested DFT energy range; missing DOS outside that range does not imply zero states.
- FHI-aims and DeePTB projected DOS use different basis/projection conventions and should be compared qualitatively rather than as exact orbital populations.
- `compare_frontiers.py` uses a spectral heuristic to identify host-like VBM/CBM bands in defect systems. Localization-sensitive quantities such as IPR or projected character remain preferable for ambiguous defect states.

## Python dependencies

- Python 3
- NumPy
- ASE
- PyTorch
- Matplotlib
- DeePTB and its runtime dependencies

Use the same DeePTB environment for training and evaluation to avoid checkpoint/API incompatibilities.

## NSCC workflow

For controlled NSCC comparisons, first freeze the train/validation membership with `split_train_val.py`, then keep those converted targets fixed and separate training outputs by experiment, for example:

```text
results/
    baseline_2x_equal/
    weighted_2x_w050/
```

A typical sequence is:

```bash
# Create or reproduce the train/validation split once
python preprocessing/split_train_val.py CONVERTED_DATA SPLIT_DATA \
    --val-list validation_sets.txt

# Check metadata state
python preprocessing/weight_band_regions.py SPLIT_DATA --recursive --status

# Activate equal-weight baseline
python preprocessing/weight_band_regions.py SPLIT_DATA --recursive --undo

# Run baseline training
# ... DeePTB / NSCC launch command ...

# Activate previously generated weighted metadata
python preprocessing/weight_band_regions.py SPLIT_DATA --recursive --activate-weighted

# Run weighted training with eout_weight = 0.5
# ... DeePTB / NSCC launch command ...
```

The dataset eigenvalues are identical between the two runs; only the active `info.json` energy-window metadata and the training loss configuration differ.
