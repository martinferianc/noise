from typing import Dict
import logging
from src.data import DATASETS


def get_hyperparameters(
    dataset: str, architecture: str, overwrite_hyperparameters: Dict[str, float] = {}
) -> Dict[str, float]:
    """Get the hyperparameters for the given dataset, architecture and method.

    Args:
        dataset (str): Dataset name.
        architecture (str): Architecture name.
        overwrite_hyperparameters (Dict[str, float], optional): Overwrite the default hyperparameters. Defaults to {}.
    """
    if dataset not in DATASETS:
        raise ValueError("Dataset not supported")

    hyperparameters = {}
    hyperparameters["dataset"] = dataset
    hyperparameters["architecture"] = architecture
    hyperparameters["batch_size"] = 256
    hyperparameters["valid_portion"] = 0.1
    # This is only used for the UCI datasets
    hyperparameters["test_portion"] = 0.2

    ### NOISE HYPERPARAMETERS ###
    hyperparameters["noise_types"] = []
    hyperparameters["noise_probabilities"] = []

    #### GRADIENT NOISE ####
    hyperparameters["noise_gradient_eta"] = 0.1
    hyperparameters["noise_gradient_gamma"] = 0.1

    #### TARGET NOISE ####
    hyperparameters["noise_target_label_smoothing"] = 0.1

    #### INPUT NOISE ####
    hyperparameters["noise_input_additive_uniform_sigma"] = 0.1
    hyperparameters["noise_input_multiplicative_uniform_sigma"] = 0.1
    hyperparameters["noise_input_additive_gaussian_sigma"] = 0.1
    hyperparameters["noise_input_multiplicative_gaussian_sigma"] = 0.1
    hyperparameters["noise_input_augmix_severity"] = 2
    hyperparameters["noise_input_augmix_width"] = 3
    hyperparameters["noise_input_augmix_chain_depth"] = -1
    hyperparameters["noise_input_augmix_alpha"] = 1.0
    hyperparameters["noise_input_ods_eta"] = 0.1
    hyperparameters["noise_input_ods_temperature"] = 1.0
    hyperparameters["noise_input_random_crop_horizontal_flip_crop_padding"] = 4

    #### INPUT TARGET NOISE ####
    hyperparameters["noise_input_target_mixup_alpha"] = 0.1
    hyperparameters["noise_input_target_cmixup_alpha"] = 0.1
    hyperparameters["noise_input_target_cmixup_sigma"] = 0.1

    #### MODEL NOISE ####
    hyperparameters["noise_model_shrink_and_perturb_mu"] = 0.1
    hyperparameters["noise_model_shrink_and_perturb_sigma"] = 0.1
    hyperparameters["noise_model_shrink_and_perturb_epoch_frequency"] = 4

    #### ACTIVATION NOISE ####
    hyperparameters["noise_activation_additive_uniform_sigma"] = 0.1
    hyperparameters["noise_activation_multiplicative_uniform_sigma"] = 0.1
    hyperparameters["noise_activation_additive_gaussian_sigma"] = 0.1
    hyperparameters["noise_activation_multiplicative_gaussian_sigma"] = 0.1
    hyperparameters["noise_activation_dropout_p"] = 0.3

    #### WEIGHT NOISE ####
    hyperparameters["noise_weight_additive_uniform_sigma"] = 0.1
    hyperparameters["noise_weight_multiplicative_uniform_sigma"] = 0.1
    hyperparameters["noise_weight_additive_gaussian_sigma"] = 0.1
    hyperparameters["noise_weight_multiplicative_gaussian_sigma"] = 0.1
    hyperparameters["noise_weight_dropconnect_p"] = 0.3

    ### OPTIMIZER HYPERPARAMETERS ###
    hyperparameters["gradient_norm_clip"] = 20.0  # float("inf")
    hyperparameters["lr"] = 0.001
    hyperparameters["l2"] = 0.0

    # Architectures
    if architecture == "resnet":
        hyperparameters["layers"] = [2, 2, 2, 2]
        hyperparameters["strides"] = [1, 2, 2, 2]
        hyperparameters["planes"] = [64, 128, 256, 512]

    elif architecture == "global_pooling_cnn":
        hyperparameters["planes"] = [128, 128, 128]

    elif architecture == "fc":
        if dataset == "svhn" or dataset == "rotated_svhn":
            hyperparameters["planes"] = [150, 150, 150, 150]
        else:
            hyperparameters["planes"] = [100, 100, 100, 100]

    elif architecture == "transformer":
        hyperparameters["dim"] = 100
        hyperparameters["depth"] = 6
        hyperparameters["heads"] = 8
        hyperparameters["mlp_dim"] = 1024
        hyperparameters["dim_head"] = 64

    # Dataset specific hyperparameters
    if dataset == "cifar10":
        hyperparameters["lr"] = 0.009793007161523374
        hyperparameters["l2"] = 0.0033326096576845403
        hyperparameters["epochs"] = 200

    elif dataset == "rotated_cifar10":
        hyperparameters["epochs"] = 200
        hyperparameters["gradient_norm_clip"] = 20.0

    elif dataset == "cifar100":
        hyperparameters["lr"] = 0.029523231194148087
        hyperparameters["l2"] = 0.0003643493156777225
        hyperparameters["epochs"] = 200

    elif dataset == "rotated_cifar100":
        hyperparameters["lr"] = 0.004430375245218269
        hyperparameters["l2"] = 0.001954952448425989
        hyperparameters["epochs"] = 200
        hyperparameters["gradient_norm_clip"] = 20.0

    elif dataset == "tinyimagenet":
        hyperparameters["lr"] = 0.025534903600640233
        hyperparameters["l2"] = 0.0004217286471739868
        hyperparameters["epochs"] = 200

    elif dataset == "rotated_tinyimagenet":
        hyperparameters["epochs"] = 200
        hyperparameters["gradient_norm_clip"] = 20.0

    elif dataset == "wiki_face":
        hyperparameters["lr"] = 0.0006378446944407505
        hyperparameters["l2"] = 0.0021809490801163707
        hyperparameters["epochs"] = 100
        hyperparameters["gradient_norm_clip"] = 5.0

    elif dataset == "svhn":
        hyperparameters["lr"] = 0.014361027310005589
        hyperparameters["l2"] = 8.401956591508542e-05
        hyperparameters["epochs"] = 200

    elif dataset == "rotated_svhn":
        hyperparameters["epochs"] = 200
        hyperparameters["gradient_norm_clip"] = 20.0

    elif dataset == "regression_energy":
        hyperparameters["lr"] = 0.006431172050131994
        hyperparameters["l2"] = 0.00018590843630169612
        hyperparameters["gradient_norm_clip"] = 10.0
        hyperparameters["epochs"] = 100
    elif dataset == "regression_boston":
        hyperparameters["lr"] = 0.09865659655619714
        hyperparameters["l2"] = 0.0004036288027444272
        hyperparameters["gradient_norm_clip"] = 10.0
        hyperparameters["epochs"] = 100
    elif dataset == "regression_wine":
        hyperparameters["lr"] = 0.004430375245218269
        hyperparameters["l2"] = 0.001954952448425989
        hyperparameters["gradient_norm_clip"] = 10.0
        hyperparameters["epochs"] = 100
    elif dataset == "regression_yacht":
        hyperparameters["lr"] = 0.07290097185098625
        hyperparameters["l2"] = 1.0761836425688191e-05
        hyperparameters["gradient_norm_clip"] = 10.0
        hyperparameters["epochs"] = 100
    elif dataset == "regression_concrete":
        hyperparameters["lr"] = 0.020954485953363917
        hyperparameters["l2"] = 0.0015872102062766636
        hyperparameters["gradient_norm_clip"] = 10.0
        hyperparameters["epochs"] = 100
    elif dataset == "classification_wine":
        hyperparameters["lr"] = 0.047394794846956426
        hyperparameters["l2"] = 0.00036302218164738594
        hyperparameters["epochs"] = 100
    elif dataset == "classification_toxicity":
        hyperparameters["lr"] = 0.07833734534369911
        hyperparameters["l2"] = 0.00018510549634117817
        hyperparameters["epochs"] = 100
    elif dataset == "classification_abalone":
        hyperparameters["lr"] = 0.07122472352381437
        hyperparameters["l2"] = 4.2483024646236366e-05
        hyperparameters["epochs"] = 100
    elif dataset == "classification_students":
        hyperparameters["lr"] = 0.04971067243477636
        hyperparameters["l2"] = 0.00039690664367130515
        hyperparameters["epochs"] = 100
    elif dataset == "classification_adult":
        hyperparameters["lr"] = 0.02369505229021759
        hyperparameters["l2"] = 0.0002483263596402237
        hyperparameters["epochs"] = 100

    elif dataset == "newsgroup":
        if architecture == "global_pooling_cnn":
            hyperparameters["lr"] = 0.006431172050131994
            hyperparameters["l2"] = 0.00018590843630169612
        else:
            hyperparameters["lr"] = 0.008930936237340367
            hyperparameters["l2"] = 0.00042133775222061255
        hyperparameters["epochs"] = 100
    elif dataset == "sst":
        if architecture == "global_pooling_cnn":
            hyperparameters["lr"] = 0.006431172050131994
            hyperparameters["l2"] = 0.00018590843630169612
        else:
            hyperparameters["lr"] = 0.008930936237340367
            hyperparameters["l2"] = 0.00042133775222061255
        hyperparameters["epochs"] = 100

    else:
        raise ValueError(f"Dataset {dataset} not supported")

    for key, value in overwrite_hyperparameters.items():
        if key not in hyperparameters:
            raise ValueError("Key {} not in hyperparameters".format(key))
        if isinstance(value, str) and value.lower() == "true":
            value = True
        elif isinstance(value, str) and value.lower() == "false":
            value = False
        logging.info(
            "#### Overwriting hyperparameter {} with {}, was {}".format(
                key, value, hyperparameters[key]
            )
        )
        hyperparameters[key] = value
    return hyperparameters
