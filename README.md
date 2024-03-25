# Navigating Noise: A Study of How Noise Influences Generalisation and Calibration of Neural Networks

by Martin Ferianc*, Ondrej Bohdal*, Timothy Hospedales, Miguel R. D. Rodrigues

*Equal contribution.

TLDR: Investigation of how noise perturbations impact neural network calibration and generalisation, identifying which perturbations are helpful and when.

Paper: [TMLR](https://openreview.net/forum?id=zn3fB4VVF0), Video: [YouTube](TODO)

- [Navigating Noise: A Study of How Noise Influences Generalisation and Calibration of Neural Networks](#navigating-noise-a-study-of-how-noise-influences-generalisation-and-calibration-of-neural-networks)
  - [Abstract](#abstract)
  - [Software implementation](#software-implementation)
    - [Code structure](#code-structure)
  - [Getting the code](#getting-the-code)
  - [Dependencies](#dependencies)
  - [License](#license)
  - [Citation](#citation)

## Abstract

Enhancing the generalisation abilities of neural networks (NNs) through integrating noise such as MixUp or Dropout during training has emerged as a powerful and adaptable technique. Despite the proven efficacy of noise in NN training, there is no consensus regarding which noise sources, types and placements yield maximal benefits in generalisation and confidence calibration. This study thoroughly explores diverse noise modalities to evaluate their impacts on NN's generalisation and calibration under in-distribution or out-of-distribution settings, paired with experiments investigating the metric landscapes of the learnt representations, across a spectrum of NN architectures, tasks, and datasets. Our study shows that AugMix and weak augmentation exhibit cross-task effectiveness in computer vision, emphasising the need to tailor noise to specific domains. Our findings emphasise the efficacy of combining noises and successful hyperparameter transfer within a single domain but the difficulties in transferring the benefits to other domains. Furthermore, the study underscores the complexity of simultaneously optimising for both generalisation and calibration, emphasising the need for practitioners to carefully consider noise combinations and hyperparameter tuning for optimal performance in specific tasks and datasets.

## Software implementation

All source code used to generate the results for the paper is in this repository.

### Code structure

```bash
.
├── experiments # scripts for running and numerically evaluating the experiments
├── README.md
├── requirements.txt
└── src # source code for the experiments
```

The main interface to run the training and evaluation of different configurations are the Python scripts: `experiments/optimize.py`, `experiments/train.py` and `experiments/evaluate.py` which can be run with the `--help` flag to see the available options for setting up the experiments.

The numerical results can be replicated by following the example instructions/commands in the `experiments/optimize.sh` which will find the hyperparameters for the noise perturbations and then in the `experiments/retrain.sh` which will train the models with the found hyperparameters.
Furthermore, we provide the `experiments/transfer_architectures.sh` and `experiments/transfer_datasets.sh` which will perform the transfer experiments.
The `experiments/analyse_results.ipynb` notebook can be used to process the results and generate the Figures for the experiments.

For the visualisations, there is the `experiments/visualise.sh` which will generate the visualisations for the noise perturbations. 

Directory `hpo_summaries` includes the found hyperparameters for each setup and `result_summaries` includes the results of our experiments that are used for generating the tables and Figures and `visualisation_summaries` includes the visualisations.

The `experiments/tabular_datasets_ood.ipynb` notebook can be used to discover the scaling factor for the OOD experiments on the tabular datasets.

## Getting the code

You can download a copy of all the files in this repository by cloning the [git](https://git-scm.com/) repository.
<!-- 
    git clone TODO
 -->
## Dependencies

You will need a working Python environment to run the code. Specifically we used Python 3.9.12.

The recommended way to set up your environment is through the [virtual environment](https://docs.python.org/3/library/venv.html#:~:text=A%20virtual%20environment%20is%20a,part%20of%20your%20operating%20system.) and the `./requirements.txt` which we provide in the root of the repository. 

```
python3 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
pip3 install -r requirements.txt
```

The libraries needed to run our code include: `torch`, `torchvision`, `optuna`, `matplotlib`, `numpy`, `scienceplots`, `sklearn`, `natsort`, `skimage`, `scipy`, `cv2`, `PIL`, `einops`, `torchmetrics` and `tensorflow.keras` (for text pre-processing).

## License

All source code is made available under an MIT license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See LICENSE.md for the full license text.

The manuscript text is not open source. The authors reserve the rights to the article content. If you find this work helpful, please consider citing our work.

## Citation

```
@article{ferianc2024navigating,
  title={Navigating Noise: A Study of How Noise Influences Generalisation and Calibration of Neural Networks},
  author={Ferianc, Martin and Bohdal, Ondrej and Hospedales, Timothy and Rodrigues, Miguel R D},
  journal={Transactions on Machine Learning Research},
  year={2024}
}
```


