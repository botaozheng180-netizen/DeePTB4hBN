# DeePTB4hBN

Utilities developed for the hBN defect-family DeePTB workflow, linking FHI-aims dense-grid and band-path calculations to DeePTB dataset preparation, optional energy-region weighting, model evaluation, and electronic-structure diagnostics.

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
        +-- visualization/batch_visualize_run.py
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
| `visualization/batch_visualize_run.py` | Batch-run the canonical band and/or band+DOS visualizers while importing heavy modules and loading the checkpoint only once | Train/validation run |
| `evaluation/evaluate_model.py` | Batch checkpoint evaluation against stored train/validation targets | Many structures |
| `evaluation/evaluate_band_error.py` | Band-index-resolved error analysis, including unsupervised non-core bands and optional two-checkpoint comparison | One structure, one/two checkpoints |
| `evaluation/compare_frontiers.py` | Compare model sources using DFT-defined host-like VBM/CBM/gap targets | Many structures, two sources |

## Quick start: choose mesh or path

Run commands from the repository root. Uppercase names such as `AIMS_ROOT`,
`MESH_DATA` and `MODEL.pth` are placeholders: replace them with your paths.
Use a different output root for each sampling type. The source tree may contain
both types of output; the selected source is the only one read.

Install the conversion, preprocessing and QA dependencies:

```bash
python -m pip install numpy ase matplotlib
```

Checkpoint evaluation and band/DOS prediction require your existing working
DeePTB/PyTorch environment and a trained checkpoint. This repository provides
dataset and diagnostic scripts; it does not include a training configuration or
checkpoint. `band_dos_compare.py` also imports DeePTB in ground-truth mode.

### 1. Choose the conversion route

| Selector | Input in each calculation folder | Intended use |
| --- | --- | --- |
| `--sampling mesh` | `geometry.in`, `control.in`, `Final_KS_eigenvalues.dat` | Dense post-SCF targets and mesh QA/evaluation |
| `--sampling path` | `geometry.in`, `band1*.out` | Selected band-path targets, band-line validation and plots |

`path` remains the default for old commands, but explicit selection is recommended.
The old `--eigenvalue-source postscf` and `--eigenvalue-source band` options remain
aliases for `mesh` and `path`. If both selectors are supplied, they must agree.
There is no source auto-detection or fallback. These flags select **existing
calculation output**; they do not run FHI-aims or generate new k-points.

#### Dense-grid conversion

Convert one case:

```bash
python conversion/aims_to_deeptb.py CASE_DIR -o MESH_SET --sampling mesh --band-policy adaptive-factor --band-factor 2.0
```

Convert a calculation tree, then resume it if needed:

```bash
python conversion/batch_aims_to_deeptb.py AIMS_ROOT MESH_DATA --sampling mesh --band-policy adaptive-factor --band-factor 2.0
python conversion/batch_aims_to_deeptb.py AIMS_ROOT MESH_DATA --sampling mesh --band-policy adaptive-factor --band-factor 2.0 --resume
```

Use `--ks-file NAME` if the post-SCF filename differs. `control.in` is required
to check spin/SOC settings. Dense conversion currently supports scalar,
non-spin-polarized calculations only.

#### Band-path conversion

Convert one case:

```bash
python conversion/aims_to_deeptb.py CASE_DIR -o PATH_SET --sampling path --band-policy adaptive-factor --band-factor 2.0
```

Convert a calculation tree, then resume it if needed:

```bash
python conversion/batch_aims_to_deeptb.py AIMS_ROOT PATH_DATA --sampling path --band-policy adaptive-factor --band-factor 2.0
python conversion/batch_aims_to_deeptb.py AIMS_ROOT PATH_DATA --sampling path --band-policy adaptive-factor --band-factor 2.0 --resume
```

Use `--band-glob 'band1*.out'` to customize path-file selection. Path segments
follow natural filename order; repeated boundaries are removed by default.
`--keep-boundary-duplicates` is available in the single-case converter. The batch
converter skips `band2*.out` cases unless `--include-spin2` explicitly allows
using only the selected channel; it does not combine spin channels.

#### Band policies and optional charge (both routes)

`adaptive-factor` strips the inferred H/B/C/N/O 1s core and keeps about
`band_factor * N_full` non-core bands, extending the selection when needed to
retain occupied states. Available-state limits still apply. Alternatives:

```bash
python conversion/aims_to_deeptb.py CASE_DIR -o MESH_ALL --sampling mesh --band-policy all-noncore
python conversion/aims_to_deeptb.py CASE_DIR -o PATH_FIXED --sampling path --band-policy fixed-noncore --noncore-bands 12
```

Both policies also work in the batch converter and with either sampling choice.
For a known neutral case add `--charge 0`; positive charge removes electrons.
When the selected window starts immediately after the inferred core, the report
records `modeled_electrons = sum(neutral valence contributions) - charge`, using
H:1, B:3, C:4, N:5, O:6. Charge omitted, unsupported chemistry or an incompatible
window gives `null`. This diagnostic field does not configure prediction-time
Fermi settings. Occupations are never summed without k-point weights to infer an
electron count. Convert different known charge states in separate batch groups.

### 2. Split, optionally weight, then inspect the selected route

At least two selected sets are required for a train/validation split:

```bash
python preprocessing/split_train_val.py MESH_DATA MESH_SPLIT --sampling mesh --val-fraction 0.2 --seed 42
python preprocessing/split_train_val.py PATH_DATA PATH_SPLIT --sampling path --val-fraction 0.2 --seed 42
```

Add `--dry-run` to preview. Splitting rejects a mixture of mesh and path sets.
The split manifest records sampling as well as structure membership. Set IDs are
local to each conversion root: use the source mappings to align physical
structures across experiments. The same seed alone does not guarantee matching
membership if the mesh and path collections contain different cases. Do not
train on a structure's mesh and validate on the same structure's path.

Optional weighting (omit `--apply` to inspect without changing metadata):

```bash
python preprocessing/weight_band_regions.py MESH_SPLIT --recursive --sampling mesh --apply
python preprocessing/weight_band_regions.py PATH_SPLIT --recursive --sampling path --apply
```

QA without a checkpoint, with or without weighting metadata:

```bash
python visualization/visualize_band_regions.py MESH_SPLIT --sampling mesh --mode selected --output RESULTS/mesh_qa
python visualization/visualize_band_regions.py PATH_SPLIT --sampling path --mode selected --output RESULTS/path_qa
python visualization/visualize_band_regions.py PATH_SPLIT --sampling path --mode full --output RESULTS/path_full_qa
```

Mesh QA uses unconnected points versus the k-point index in file order, with no
high-symmetry labels. Path QA uses cumulative reciprocal-space distance. `full`
QA reloads original path bands and is unavailable for mesh sets.

### 3. Evaluate a checkpoint on the selected route

After training in your DeePTB environment, supply the actual checkpoint:

```bash
python evaluation/evaluate_model.py --checkpoint MODEL.pth --train MESH_SPLIT/train --val MESH_SPLIT/val --sampling mesh --output RESULTS/mesh_eval --make-plots
python evaluation/evaluate_model.py --checkpoint MODEL.pth --train PATH_SPLIT/train --val PATH_SPLIT/val --sampling path --output RESULTS/path_eval --make-plots
```

For single-set, band-index-resolved errors:

```bash
python evaluation/evaluate_band_error.py --checkpoint MODEL.pth --set MESH_DATA/set.000000 --mode supervised --output RESULTS/mesh_error
python evaluation/evaluate_band_error.py --checkpoint MODEL.pth --set PATH_DATA/set.000000 --mode full --output RESULTS/path_error
```

Mesh error evaluation supports `supervised` mode only. `full` reloads original
non-core path bands. Neither error plot turns the mesh into a band path.

For host-like edge/gap comparison between two checkpoints or training runs:

```bash
python evaluation/compare_frontiers.py --source-a RUN_A --source-b RUN_B --train MESH_SPLIT/train --val MESH_SPLIT/val --sampling mesh --output RESULTS/mesh_frontiers --make-plots
python evaluation/compare_frontiers.py --source-a RUN_A --source-b RUN_B --train PATH_SPLIT/train --val PATH_SPLIT/val --sampling path --output RESULTS/path_frontiers --make-plots
```

Frontier comparisons use a host-edge heuristic over the supplied points; mesh
and path extrema may differ. This tool assumes neutral charge by default: supply
`--charge-map` for charged cases. Reused evaluation predictions must come from
the exact same structures, k-points and set mapping.

### 4. Use path datasets for band-line and band/DOS plots

```bash
python visualization/band_plot.py --checkpoint MODEL.pth --set PATH_DATA/set.000000 --mode supervised --output RESULTS/path_band
python visualization/band_dos_compare.py --set PATH_DATA/set.000000 --mode full --plot-content ground-truth --dos-mode all --output RESULTS/path_dos
python visualization/batch_visualize_run.py --run-dir PATH_RUN --backend band --mode supervised --dry-run
python visualization/batch_visualize_run.py --run-dir PATH_RUN --backend both --mode full
```

`PATH_RUN` input JSON must reference **path** train/validation roots. Batch
visualization checks every discovered set in the selected splits before limiting
cases or loading a model; any mesh set causes an error, also in `--dry-run`.
A model trained on mesh data can be plotted on a separate converted path dataset
using the single-case commands with its checkpoint. Ground-truth DOS plotting
also needs the original FHI-aims DOS files; see the detailed commands below.

## Sampling metadata, outputs and resume rules

| File | Shape / meaning |
| --- | --- |
| `xdat.traj` | One ASE structure |
| `kpoints.npy` | `(Nk, 3)`, fractional reciprocal coordinates |
| `eigenvalues.npy` | `(1, Nk, Nb)`, selected energies in native eV gauge |
| `occupations.npy` | `(1, Nk, Nb)`, occupations paired with the selected states |
| `info.json` | DeePTB training metadata |
| `conversion_report.json` | `kpoint_sampling`, source, hashes, band policy and optional electron count |

The leading `1` is a frame axis, not spin. Batch output is `set.XXXXXX` plus
`manifest.json` and `manifest.csv`. Keep the report and occupations with the
arrays; splitting copies or links the complete set directory.

Downstream collection tools (`split_train_val.py`, `weight_band_regions.py`,
`visualize_band_regions.py`, `evaluate_model.py`, `compare_frontiers.py`) accept
`--sampling mesh`, `--sampling path`, or `--sampling all` (default). No matches
is an error. `all` preserves legacy collection behavior except mixed sampling
is rejected when splitting. Single-set plotting reads sampling from the report.
Legacy reports without sampling/source metadata are treated as path datasets;
unknown or contradictory metadata is rejected. Do not remove mesh reports.

Use sibling source/output roots. Batch output cannot be nested inside the
source tree, and an existing mesh collection cannot be reused as a path output
(or vice versa), even with `--overwrite`. Choose a fresh output root instead.
`--resume` and `--overwrite` are mutually exclusive. Resume checks source
identity/type, charge, policy, input SHA-256 hashes and required output presence.
Source changes or changed set ordering are rejected. Hashless old reports need
regeneration. Conversion failures return a nonzero exit code and are recorded
in the manifests.

The supported dense format contains a fractional-coordinate header
`# k-point in recip. lattice units: kx ky kz` followed by consecutive
`state occupation eigenvalue` rows in eV. Fortran D exponents are accepted.
The parser preserves source order and rejects incomplete/inconsistent blocks,
duplicate states/reciprocal points, nonfinite values and recognized multi-spin
layouts. Declared counts are checked when present. Entire missing blocks cannot
be detected without counts; independently verify FHI-aims completion.
It does not reconstruct the grid from `control.in`. See the
[FHI-aims post-SCF output documentation](https://www.fhi-aims.org/uploads/manual/Ch3/S56.html).

Errors and weighting diagnostics give equal weight to stored k-points. This
repository does not add Brillouin-zone quadrature weights or mesh DOS integration.
An unweighted symmetry-reduced sample is not a BZ integral.

Run the local checks (also run in GitHub Actions):

```bash
python -m unittest discover -s tests -v
python -m compileall -q conversion preprocessing evaluation visualization
```

Tests use synthetic spectra and cover both source routes, selection conflicts,
no fallback, occupations, charge, resume, sampling filters, split integrity and
mesh QA/guards. A real NSCC output and trained checkpoint are not included;
validate representative production input and inference in your own environment.


## Dataset splitting options


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
python visualization/visualize_band_regions.py PATH_DATA --sampling path --mode full
```

Full mode reloads `band1*.out`, strips the inferred 1s core bands, and verifies that the converted target is the expected prefix of the reconstructed non-core spectrum.

By default the visualizer reads the canonical `info.weighted.json`. To inspect the metadata currently active in DeePTB:

```bash
python visualization/visualize_band_regions.py DATA \
    --mode selected \
    --window-source active
```

The plots use:

- black lines for path bands, or unconnected black points for mesh spectra;
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

Use `--sampling mesh` or `--sampling path` on batch evaluation and frontier
comparison commands to select one route. Band-line and band/DOS commands
below require path data.

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
    --set PATH_DATA/set.000000 \
    --output RESULTS/band_plot \
    --mode supervised
```

Full available non-core spectrum:

```bash
python visualization/band_plot.py \
    --checkpoint RUN/checkpoint/nnsk.epN.pth \
    --set PATH_DATA/set.000000 \
    --output RESULTS/band_plot_full \
    --mode full
```

`--mode full` uses `source_case` in `conversion_report.json` unless `--aims-dir` is supplied.

### Compare band structure and DOS

Ground truth only:

```bash
python visualization/band_dos_compare.py \
    --set PATH_DATA/set.000000 \
    --output RESULTS/band_dos_dft \
    --plot-content ground-truth \
    --mode full \
    --dos-mode all
```

DeePTB prediction only:

```bash
python visualization/band_dos_compare.py \
    --checkpoint RUN/checkpoint/nnsk.epN.pth \
    --set PATH_DATA/set.000000 \
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
    --set PATH_DATA/set.000000 \
    --output RESULTS/band_dos_compare \
    --plot-content comparison \
    --mode full \
    --dos-mode all \
    --kmesh 30 30 1 \
    --sigma 0.10
```

`--dos-mode total` plots only total DOS; `--dos-mode all` adds one species-projected panel for every species present in the structure.

The script prefers the ordinary FHI-aims `KS_DOS_total.dat` and `<species>_l_proj_dos.dat` files so the DOS and FHI-aims bands share the same energy reference. If the DOS calculation covers a narrower energy window than the plotted bands, the script warns rather than treating missing DOS as an absence of states.

### Batch-visualize a training run

The batch visualizer reads the DeePTB run input, resolves its train/validation dataset roots, auto-detects the checkpoint, and calls the canonical single-case plotting scripts in one Python process. PyTorch/DeePTB are imported once and the checkpoint is loaded once, which avoids repeated startup overhead on NSCC.

Band plots for all train + validation structures:

```bash
python visualization/batch_visualize_run.py \
    --run-dir RUN \
    --backend band \
    --mode full
```

Band + DOS plots:

```bash
python visualization/batch_visualize_run.py \
    --run-dir RUN \
    --backend both \
    --mode full
```

Test one validation case first:

```bash
python visualization/batch_visualize_run.py \
    --run-dir RUN \
    --backend band \
    --mode full \
    --splits val \
    --limit 1
```

Use `--checkpoint` or `--input-json` to override auto-detection. The DOS backend also accepts `--dos-mode`, `--kmesh`, `--sigma`, and `--plot-content`.

The default output is:

```text
RUN/visualization/
    band/
        train/
        val/
    dos/
        train/
        val/
    flat_png/
    batch_visualization_summary.csv
    batch_visualization_failures.csv
    batch_visualization_manifest.json
```

`flat_png/` collects copies of generated PNGs with split/set/backend prefixes for quick browsing. Scientific plotting logic remains in `band_plot.py` and `band_dos_compare.py`; the batch script only resolves the run and orchestrates repeated calls.

### Inspect error versus band index

```bash
python evaluation/evaluate_band_error.py \
    --checkpoint RUN/checkpoint/nnsk.epN.pth \
    --set PATH_DATA/set.000000 \
    --output RESULTS/band_error_epN \
    --mode full
```

Compare two checkpoints:

```bash
python evaluation/evaluate_band_error.py \
    --checkpoint RUN_A/checkpoint/nnsk.epN.pth \
    --checkpoint2 RUN_B/checkpoint/nnsk.epM.pth \
    --set PATH_DATA/set.000000 \
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
- Path-band visualization uses the hBN `M -> Gamma -> K -> M` convention; mesh QA uses file-order indices.
- `weight_band_regions.py` detects a spectral separation, not a chemically exact 2s projector. The low-density region is a detection aid; the selected `emin` is the actual weighting boundary.
- Weighted and unweighted validation losses are not directly comparable when different `eout_weight` settings are used. Preserve an unweighted evaluation route when comparing training strategies.
- `band_dos_compare.py` keeps FHI-aims in its native band/DOS energy gauge and rigidly aligns DeePTB to it when `--align loss` is used.
- FHI-aims DOS files cover only the requested DFT energy range; missing DOS outside that range does not imply zero states.
- FHI-aims and DeePTB projected DOS use different basis/projection conventions and should be compared qualitatively rather than as exact orbital populations.
- `compare_frontiers.py` uses a spectral heuristic to identify host-like VBM/CBM bands in defect systems. Localization-sensitive quantities such as IPR or projected character remain preferable for ambiguous defect states.

## Python dependencies

- Python 3.10 or newer
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


After path training, batch visualization can be launched from the path run metadata:

```bash
python visualization/batch_visualize_run.py \
    --run-dir PATH_RUN \
    --backend band \
    --mode full
```

Use `--backend both` when the band+DOS diagnostics are also required.

For a mesh experiment use mesh QA and stored-target evaluation instead:

```bash
python visualization/visualize_band_regions.py MESH_SPLIT --sampling mesh --mode selected --output RESULTS/mesh_qa
python evaluation/evaluate_model.py --checkpoint MODEL.pth --train MESH_SPLIT/train --val MESH_SPLIT/val --sampling mesh --output RESULTS/mesh_eval --make-plots
```
