# Experiments of the paper

One notebook per experiment. Each notebook states the method as in the paper, runs it and writes its outputs to `outputs/`.

| Paper | Notebook | Run |
| --- | --- | --- |
| Model | [`../un-CNN.ipynb`](../un-CNN.ipynb) | any machine |
| Datasets, main benchmark, permutation tests | [`01_main_benchmark.ipynb`](01_main_benchmark.ipynb), one dataset per run | `slurm/main_benchmark.sh`: 3 array tasks, 16 CPUs, 48 GB |
| Figure 2 values and results statements | [`02_main_benchmark_results.ipynb`](02_main_benchmark_results.ipynb) | after the three runs of 01 |
| Table I: Params., Dim., Input prep., Feat. ext., Total | [`03_encoder_timing.ipynb`](03_encoder_timing.ipynb) | `slurm/encoder_timing.sh`: 2 CPUs, one thread |
| Table I: Head fit | [`04_head_timing.ipynb`](04_head_timing.ipynb) | `slurm/head_timing.sh`, after the NKI run of 01 |
| Table II, Figure 3: architecture ablation | [`05_architecture_ablation.ipynb`](05_architecture_ablation.ipynb) | `slurm/architecture_ablation.sh`: 16 CPUs |
| Integrated gradients | [InesGP/untrained-CNN-saliency](https://github.com/InesGP/untrained-CNN-saliency) | |

Shared code: `cohorts.py` builds the subjects, labels and inputs of each dataset, `encoders.py` the frozen feature extractors, `heads.py` the linear head, `config.py` reads the paths and `machine.py` records the hardware of a timing run.

## Setup

```bash
git clone https://github.com/epics-lab/untrained-cnn.git
cd untrained-cnn
pip install -e .
pip install -r experiments/requirements.txt
pip install --no-deps monai==1.4.0   # its PyPI metadata requires numpy < 2
cp experiments/paths.example.env experiments/paths.env   # then edit
```

The paper runs used the Rorqual cluster of the Digital Research Alliance of Canada, nodes with two AMD EPYC 9654 processors, AlmaLinux 9.8, Python 3.11.5, PyTorch 2.10.0, MONAI 1.4.0, NumPy 2.4.2 and scikit-learn 1.7.1.

## Data

No data and no subject level output is in this repository; access follows the data use agreement of each study.

- NKI and HBN: [Reproducible Brain Charts](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11888297/), with its FreeSurfer v6.0.1 surface statistics.
- PPMI: [ppmi-info.org](https://www.ppmi-info.org/), with its FreeSurfer v7.3.2 aparc tables.

Model inputs per subject:

- affine volume: HD-BET skull stripping, then TurboPrep with affine registration to the MNI152 ICBM 2009c nonlinear symmetric template, `<AFFINE_DIR>/<subject>/normalized.nii.gz`;
- CAT12 gray matter map, `mwp1*.nii`, for AnatCL;
- rigid volume, for the architecture ablation only: TurboPrep with `-r r` on the scans without HD-BET, `<RIGID_DIR>/<subject>/normalized.nii.gz`.

`paths.example.env` lists every path with the expected file format.

## Checkpoints

| Model | Source | Path variable |
| --- | --- | --- |
| AnatCL | [carloalbertobarbano/AnatCL](https://github.com/carloalbertobarbano/AnatCL), installed by `requirements.txt`; the five fold checkpoints of each descriptor | `ANATCL_WEIGHTS_DIR`, with `weights_{global,local}_fold{0..4}.pth` |
| BrainIAC | [AIM-KannLab/BrainIAC](https://github.com/AIM-KannLab/BrainIAC) | `BRAINIAC_CKPT` |
| 3D-Neuro-SimCLR | [emilykaczmarek/3D-Neuro-SimCLR](https://github.com/emilykaczmarek/3D-Neuro-SimCLR) | `SIMCLR_CKPT` |
| SwinBrain | [MAI-Lab-West-China-Hospital/SwinBrain](https://github.com/MAI-Lab-West-China-Hospital/SwinBrain) | `SWINBRAIN_CKPT` |

The un-CNN needs no checkpoint. No checkpoint is redistributed here.

## Running

Interactively, open a notebook from `experiments/` and set the parameters in its first code cell. On SLURM, submit from `experiments/` with the Python environment active, or with `VENV` pointing to it:

```bash
cd experiments
sbatch --account=<account> slurm/main_benchmark.sh   # 01 for PPMI, NKI and HBN
sbatch --account=<account> slurm/encoder_timing.sh   # 03
sbatch --account=<account> slurm/head_timing.sh      # 04, after the NKI task of the benchmark
sbatch --account=<account> slurm/architecture_ablation.sh   # 05
```

Each job saves the executed notebook with its outputs in `outputs/executed/`. `outputs/` also holds subject level features and predictions, so git ignores it.
