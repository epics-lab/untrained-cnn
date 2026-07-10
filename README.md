[![Untrained CNN models available](https://img.shields.io/badge/🤗%20Hugging%20Face-Untrained%20CNN%20models-yellow)](https://huggingface.co/epics-lab/untrained-cnn)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.06.07.730652-b31b1b.svg)](https://www.biorxiv.org/content/10.64898/2026.06.07.730652v1)

# untrained-cnn

This repository provides an implementation of **un-CNN**, an untrained 3D convolutional neural network used as a fixed feature extractor for structural brain MRI.

The model is **never trained**. Its weights are initialized once with a fixed random seed and kept frozen. Brain MRI volumes are passed through the untrained CNN to obtain subject-level feature vectors, which are then evaluated using downstream machine-learning models such as Random Forests.

## Installation

```bash
pip install git+https://github.com/epics-lab/untrained-cnn.git
```

This installs the `uncnn` module along with its dependencies (numpy, scipy, nibabel, torch). No cloning required.

## Usage

```python
import uncnn

# one scan
feats = uncnn.extract_features("subject01.nii.gz")          # -> shape (1, 9664)

# many scans
import glob
feats = uncnn.extract_features(glob.glob("data/*.nii.gz"))  # -> shape (N, 9664)
```

Inputs are any NIfTI volume (`.nii` / `.nii.gz`); each is resized to the MNI152 2mm grid (91x109x91) and converted to a 3-channel image automatically. The output is a NumPy array with one feature row per subject — feed it to whatever downstream model you like (e.g. a Random Forest).

### Options

```python
uncnn.extract_features(paths,
                       config="robust",   # "robust" (default) or "ranksobel"
                       batch_size=8,       # default
                       device=None,        # "cpu" or "cuda"; auto-detected if omitted
                       seed=0)             # reproducible random weights

model = uncnn.load_model(seed=0)          # raw frozen model, if you want it directly
```

`config="robust"` is the best overall age configuration; `config="ranksobel"` (clip 2/98) is the best sex-accuracy configuration.

By default the device is chosen automatically — CUDA if a GPU is available, otherwise CPU. Both work; CPU runs fine for small batches, and you can force either with `device="cpu"` or `device="cuda"`.

## Input preprocessing

Inputs should be T1-weighted MRI scans preprocessed with [TurboPrep](https://github.com/LemuelPuglisi/turboprep) and registered to MNI152 space.

Run TurboPrep on each scan:

```bash
turboprep $T1_FILE $OUTPUT_DIR $T1_TEMPLATE -m t1 -r r
```

- `$T1_FILE` — path to a T1 MRI scan in `.nii.gz` format.
- `$OUTPUT_DIR` — output directory. TurboPrep writes `normalized.nii.gz` (the preprocessed scan) and `mask.nii.gz` (the brain mask). The output filenames are always the same, so give each input its own directory.
- `$T1_TEMPLATE` — path to the MNI152 ICBM non-linear symmetric 2009c template ([download](https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/)), e.g. `mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c.nii`.
- `-m t1` — use the T1 modality for intensity normalization.
- `-r r` — use rigid registration to the template instead of the default affine.

Pass the resulting `normalized.nii.gz` files to `uncnn.extract_features`.

## Repository structure

| File | Description |
| --- | --- |
| `uncnn.py` | The installable module: final un-CNN architecture, preprocessing, and the `extract_features` API. |
| `pyproject.toml` | Packaging metadata that makes the repo pip-installable. |
| `un-CNN.ipynb` | Final un-CNN architecture and preprocessing pipeline used for the main results. Start here if you want to read the final model. |
| `Untrained CNN Project.ipynb` | Development notebook containing architecture optimization and ablation experiments. |
| `LICENSE` | MIT license. |

## Final un-CNN architecture

The final model is provided in `uncnn.py` (and mirrored in `un-CNN.ipynb`, which isolates the final architecture and preprocessing pipeline from the optimization steps).

The final un-CNN uses:

- fixed random weights with seed 0,
- 3D convolutional feature extraction,
- multi-block feature aggregation,
- first-order spatial pooling,
- covariance pooling,
- three-channel structural MRI input preprocessing.

No CNN training, fine-tuning, or backpropagation is performed. Each scan yields a 9664-dimensional feature vector.

## Architecture optimization

The development notebook `Untrained CNN Project.ipynb` contains the optimization steps used to evaluate different untrained 3D CNN design choices for brain MRI feature extraction.

Random-weight CNN features are extracted from structural MRI scans and evaluated using identical downstream Random Forest models for prediction tasks such as:

- sex classification,
- age regression,
- BMI regression.

## Citation

If you use this repository, please cite the associated preprint:

```bibtex
@article{encin2026untrained,
  title={Untrained Convolutional Neural Networks as Feature Extractors for Structural MRI},
  author={Encin, Arel and Gonzalez Pepe, Ines and Chatelain, Yohan and Dickie, Erin and Glatard, Tristan},
  journal={bioRxiv},
  pages={2026.06.07.730652},
  year={2026},
  publisher={Cold Spring Harbor Laboratory},
  doi={10.64898/2026.06.07.730652}
}
```

## License

This repository is released under the MIT License.
