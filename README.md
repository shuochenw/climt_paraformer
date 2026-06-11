# climt-paraformer

**Stable Emulation of Convective Parameterization using a Temporal Memory-aware Transformer**

`climt-paraformer` is a research codebase for building a stable neural emulator of convective parameterization in a column climate model. The project focuses on replacing an expensive convection scheme with a temporal memory-aware Transformer that can reproduce offline parameterization tendencies while remaining stable when coupled back into the climate model for online integration.

The repository includes code for climate-column data generation, preprocessing, model definition, offline evaluation, online testing, and plotting diagnostics.

## Project Goals

- Emulate convective parameterization with a Transformer-based surrogate.
- Use temporal memory so the emulator can account for recent atmospheric history instead of treating each state independently.
- Evaluate both offline accuracy and online stability.
- Support reproducible data generation from a column model with slab setup.
- Provide analysis notebooks for preprocessing, validation, and visualization.

## Repository Structure

| Path | Description |
| --- | --- |
| `column_code_with_slab/` | Climate-column model setup and data generation code. This folder contains the model configuration, slab setup, and scripts used to produce training and testing trajectories. |
| `preprocessing.ipynb` | Converts raw generated climate-column output into model-ready arrays. This includes cleaning, variable selection, normalization, windowing, and train/validation/test preparation. |
| `models.py` | Defines the neural emulator architecture, including the temporal memory-aware Transformer components used by `climt-paraformer`. |
| `offline.ipynb` | Offline evaluation notebook. Tests the trained emulator against held-out target tendencies or parameterization outputs without coupling it back into the climate model. |
| `online.py` | Online testing script. Couples the trained emulator into the column climate model and evaluates whether the parameterization surrogate remains stable during prognostic integration. |
| `online_plots.ipynb` | Visualization notebook for online experiments, including trajectory diagnostics, stability checks, and comparison plots. |

## Workflow

1. Generate climate-column simulations using `column_code_with_slab/`.
2. Run `preprocessing.ipynb` to create normalized model inputs and targets.
3. Train or load the Paraformer emulator defined in `models.py`.
4. Use `offline.ipynb` to evaluate pointwise and sequence-level offline performance.
5. Run `online.py` to test the emulator in a coupled online setting.
6. Use `online_plots.ipynb` to inspect online trajectories and diagnostics.

## Model Overview

The Paraformer emulator is designed for convective parameterization, where stability depends not only on instantaneous atmospheric state but also on temporal context. The model uses a Transformer backbone augmented with memory-aware temporal structure so that predictions can reflect recent evolution in the atmospheric column.

At a high level, the emulator maps climate-column state variables to convective tendencies or parameterization outputs:

```text
Atmospheric column state history -> Temporal memory-aware Transformer -> Convective parameterization output
```

The offline setting measures how well the emulator matches target outputs from the original parameterization. The online setting tests whether those predictions remain physically and numerically stable when inserted back into the model loop.

## Getting Started

Create an environment with the scientific Python stack and PyTorch:

```bash
conda create -n climt-paraformer python=3.10
conda activate climt-paraformer
pip install numpy scipy pandas matplotlib xarray netcdf4 jupyter torch
```

Additional dependencies may be required by the column model code in `column_code_with_slab/`.

## Running the Main Experiments

Preprocess generated data:

```bash
jupyter notebook preprocessing.ipynb
```

Run offline evaluation:

```bash
jupyter notebook offline.ipynb
```

Run online testing:

```bash
python online.py
```

Plot online diagnostics:

```bash
jupyter notebook online_plots.ipynb
```

## Outputs

Typical outputs include:

- Preprocessed model-ready datasets.
- Offline prediction metrics and diagnostic figures.
- Online coupled-model trajectories.
- Stability and drift diagnostics.
- Comparison plots between the original convective parameterization and the emulator.

## Notes

This repository is intended for research use. Online climate-model emulation can be sensitive to normalization, temporal context length, coupling frequency, and numerical stability checks. Offline skill is necessary but not sufficient; online tests in `online.py` are the primary stability test.

## Citation

If you use this code, please cite the project as:

```bibtex
@misc{climt_paraformer,
  title = {climt-paraformer: Stable Emulation of Convective Parameterization using a Temporal Memory-aware Transformer},
  author = {Shuochen},
  year = {2026},
  note = {Research code}
}
```
