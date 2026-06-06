[![Untrained CNN models available](https://img.shields.io/badge/🤗%20Hugging%20Face-Untrained%20CNN%20models-yellow)](https://huggingface.co/epics-lab/untrained-cnn)

# untrained-cnn

This repository provides an implementation of **un-CNN**, an untrained 3D convolutional neural network used as a fixed feature extractor for structural brain MRI.

The model is **never trained**. Its weights are initialized once with a fixed random seed and kept frozen. Brain MRI volumes are passed through the untrained CNN to obtain subject-level feature vectors, which are then evaluated using downstream machine-learning models such as Random Forests.

## Repository structure

| File | Description |
|---|---|
| [`un-CNN.ipynb`](./un-CNN.ipynb) | Final un-CNN architecture and preprocessing pipeline used for the main results. Start here if you want to use the final model. |
| [`Untrained CNN Project.ipynb`](./Untrained%20CNN%20Project.ipynb) | Development notebook containing architecture optimization and ablation experiments. |
| [`LICENSE`](./LICENSE) | MIT license. |

## Final un-CNN architecture

The final model is provided in [`un-CNN.ipynb`](./un-CNN.ipynb). This notebook isolates the final architecture and preprocessing pipeline from the optimization steps.

The final un-CNN uses:

- fixed random weights with seed 0,
- 3D convolutional feature extraction,
- multi-block feature aggregation,
- first-order spatial pooling,
- covariance pooling,
- three-channel structural MRI input preprocessing.

No CNN training, fine-tuning, or backpropagation is performed.

## Architecture optimization

The development notebook [`Untrained CNN Project.ipynb`](./Untrained%20CNN%20Project.ipynb) contains the optimization steps used to evaluate different untrained 3D CNN design choices for brain MRI feature extraction.

Random-weight CNN features are extracted from structural MRI scans and evaluated using identical downstream Random Forest models for prediction tasks such as:

- sex classification,
- age regression,
- BMI regression.

## Hugging Face mirror

The repository is also available on Hugging Face:

[https://huggingface.co/epics-lab/untrained-cnn](https://huggingface.co/epics-lab/untrained-cnn)

## Citation

If you use this repository, please cite the associated project/preprint when available.

## License

This repository is released under the MIT License.
