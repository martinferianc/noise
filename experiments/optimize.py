import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from typing import Literal

import numpy as np
import optuna
import torch
from evaluate import evaluate_model

import src.utils as utils
from src.training import train
from src.training.hyperparameters import get_hyperparameters

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append("../")


def sample_configuration(
    trial,
    hyperparameters,
    mode: Literal[
        "vanilla",
        "batch",
        "target_smoothing",
        "model_sp",
        "input_additive_gaussian",
        "input_additive_uniform",
        "input_multiplicative_gaussian",
        "input_multiplicative_uniform",
        "input_ods",
        "input_augmix",
        "input_target_mixup",
        "input_target_cmixup",
        "activation_additive_gaussian",
        "activation_additive_uniform",
        "activation_multiplicative_gaussian",
        "activation_multiplicative_uniform",
        "weight_additive_gaussian",
        "weight_additive_uniform",
        "weight_multiplicative_gaussian",
        "weight_multiplicative_uniform",
        "weight_dropconnect",
        "gradient_gaussian",
        "activation_dropout",
    ] = ["vanilla"],
):
    """Sample a configuration for a trial."""
    if len(mode) > 1:
        noise_types = []
        hps_dictionary = {}
        for m in mode:
            if m == "vanilla":
                current_hp_dict = {
                    "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
                    "l2": trial.suggest_float("l2", 1e-7, 1e-1, log=True),
                    "noise_types": ["none"],
                    "noise_probabilities": [0.0],
                }
            elif m == "batch":
                current_hp_dict = {
                    "batch_size": trial.suggest_categorical(
                        "batch_size", [32, 64, 128, 256, 512, 1024]
                    ),
                    "noise_types": ["none"],
                    "noise_probabilities": [0.0],
                }
            elif m == "gradient_gaussian":
                current_hp_dict = {
                    "noise_gradient_eta": trial.suggest_float(
                        "noise_gradient_eta", 0.0, 1.0
                    ),
                    "noise_gradient_gamma": trial.suggest_float(
                        "noise_gradient_gamma", 0.0, 1.0
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["gradient_gaussian"],
                }
            elif m == "target_smoothing":
                current_hp_dict = {
                    "noise_target_label_smoothing": trial.suggest_float(
                        "noise_target_label_smoothing", 0.0, 0.25
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["target_label_smoothing"],
                }
            elif m == "model_sp":
                current_hp_dict = {
                    "noise_model_shrink_and_perturb_mu": trial.suggest_float(
                        "noise_model_shrink_and_perturb_mu", 0.0, 1.0
                    ),
                    "noise_model_shrink_and_perturb_sigma": trial.suggest_float(
                        "noise_model_shrink_and_perturb_sigma", 1e-7, 0.001, log=True
                    ),
                    "noise_model_shrink_and_perturb_epoch_frequency": trial.suggest_int(
                        "noise_model_shrink_and_perturb_epoch_frequency", 1, 20
                    ),
                    "noise_types": ["model_shrink_and_perturb"],
                    "noise_probabilities": [1.0],
                }
            elif m == "input_additive_gaussian":
                current_hp_dict = {
                    "noise_input_additive_gaussian_sigma": trial.suggest_float(
                        "noise_input_additive_gaussian_sigma", 1e-4, 1e-1, log=True
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["input_additive_gaussian"],
                }
            elif m == "input_multiplicative_gaussian":
                current_hp_dict = {
                    "noise_input_multiplicative_gaussian_sigma": trial.suggest_float(
                        "noise_input_multiplicative_gaussian_sigma",
                        1e-4,
                        1e-1,
                        log=True,
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["input_multiplicative_gaussian"],
                }
            elif m == "input_additive_uniform":
                current_hp_dict = {
                    "noise_input_additive_uniform_sigma": trial.suggest_float(
                        "noise_input_additive_uniform_sigma", 1e-4, 1e-1, log=True
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["input_additive_uniform"],
                }
            elif m == "input_multiplicative_uniform":
                current_hp_dict = {
                    "noise_input_multiplicative_uniform_sigma": trial.suggest_float(
                        "noise_input_multiplicative_uniform_sigma", 1e-4, 1e-1, log=True
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["input_multiplicative_uniform"],
                }
            elif m == "input_ods":
                current_hp_dict = {
                    "noise_input_ods_temperature": trial.suggest_float(
                        "noise_input_ods_temperature", 0.5, 5.0, log=True
                    ),
                    "noise_input_ods_eta": trial.suggest_float(
                        "noise_input_ods_eta", 1e-4, 1e-1, log=True
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["input_ods"],
                }
            elif m == "input_augmix":
                current_hp_dict = {
                    "noise_input_augmix_alpha": trial.suggest_float(
                        "noise_input_augmix_alpha", 0.0, 1.0
                    ),
                    "noise_input_augmix_severity": trial.suggest_int(
                        "noise_input_augmix_severity", 1, 10
                    ),
                    "noise_input_augmix_width": trial.suggest_int(
                        "noise_input_augmix_width", 1, 5
                    ),
                    "noise_input_augmix_chain_depth": trial.suggest_categorical(
                        "noise_input_augmix_chain_depth", [-1, 1, 2, 3]
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["input_augmix"],
                }
            elif m == "input_target_mixup":
                current_hp_dict = {
                    "noise_input_target_mixup_alpha": trial.suggest_float(
                        "noise_input_target_mixup_alpha", 0.0, 1.0
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["input_target_mixup"],
                }
            elif m == "input_target_cmixup":
                current_hp_dict = {
                    "noise_input_target_cmixup_alpha": trial.suggest_float(
                        "noise_input_target_cmixup_alpha", 0.0, 1.0
                    ),
                    "noise_input_target_cmixup_sigma": trial.suggest_float(
                        "noise_input_target_cmixup_sigma", 1e-4, 1e2, log=True
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["input_target_cmixup"],
                }
            elif m == "input_random_crop_horizontal_flip":
                current_hp_dict = {
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["input_random_crop_horizontal_flip"],
                }
            elif m == "activation_additive_gaussian":
                current_hp_dict = {
                    "noise_activation_additive_gaussian_sigma": trial.suggest_float(
                        "noise_activation_additive_gaussian_sigma", 1e-4, 1e-1, log=True
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["activation_additive_gaussian"],
                }
            elif m == "activation_multiplicative_gaussian":
                current_hp_dict = {
                    "noise_activation_multiplicative_gaussian_sigma": trial.suggest_float(
                        "noise_activation_multiplicative_gaussian_sigma",
                        1e-4,
                        1e-1,
                        log=True,
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["activation_multiplicative_gaussian"],
                }
            elif m == "activation_additive_uniform":
                current_hp_dict = {
                    "noise_activation_additive_uniform_sigma": trial.suggest_float(
                        "noise_activation_additive_uniform_sigma", 1e-4, 1e-1, log=True
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["activation_additive_uniform"],
                }
            elif m == "activation_multiplicative_uniform":
                current_hp_dict = {
                    "noise_activation_multiplicative_uniform_sigma": trial.suggest_float(
                        "noise_activation_multiplicative_uniform_sigma",
                        1e-4,
                        1e-1,
                        log=True,
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["activation_multiplicative_uniform"],
                }
            elif m == "activation_dropout":
                current_hp_dict = {
                    "noise_activation_dropout_p": trial.suggest_float(
                        "noise_activation_dropout_p", 0.0, 1.0
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["activation_dropout"],
                }
            elif m == "weight_additive_gaussian":
                current_hp_dict = {
                    "noise_weight_additive_gaussian_sigma": trial.suggest_float(
                        "noise_weight_additive_gaussian_sigma", 1e-4, 1e-1, log=True
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["weight_additive_gaussian"],
                }
            elif m == "weight_multiplicative_gaussian":
                current_hp_dict = {
                    "noise_weight_multiplicative_gaussian_sigma": trial.suggest_float(
                        "noise_weight_multiplicative_gaussian_sigma",
                        1e-4,
                        1e-1,
                        log=True,
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["weight_multiplicative_gaussian"],
                }
            elif m == "weight_additive_uniform":
                current_hp_dict = {
                    "noise_weight_additive_uniform_sigma": trial.suggest_float(
                        "noise_weight_additive_uniform_sigma", 1e-4, 1e-1, log=True
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["weight_additive_uniform"],
                }
            elif m == "weight_multiplicative_uniform":
                current_hp_dict = {
                    "noise_weight_multiplicative_uniform_sigma": trial.suggest_float(
                        "noise_weight_multiplicative_uniform_sigma",
                        1e-4,
                        1e-1,
                        log=True,
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["weight_multiplicative_uniform"],
                }
            elif m == "weight_dropconnect":
                current_hp_dict = {
                    "noise_weight_dropconnect_p": trial.suggest_float(
                        "noise_weight_dropconnect_p", 0.0, 1.0
                    ),
                    "noise_probabilities": [
                        trial.suggest_float("noise_probabilities", 0.0, 1.0)
                    ],
                    "noise_types": ["weight_dropconnect"],
                }
            

            noise_types.append(current_hp_dict["noise_types"][0])
            hps_dictionary.update(current_hp_dict)

        hps_dictionary["noise_probabilities"] = [
            trial.suggest_float("noise_probabilities_" + str(idx), 0.0, 1.0)
            for idx, m in enumerate(mode)
        ]
        for idx, noise_type in enumerate(hps_dictionary["noise_types"]):
            if noise_type == "none":
                hps_dictionary["noise_probabilities"][idx] = 0.0
            elif noise_type == "model_shrink_and_perturb":
                hps_dictionary["noise_probabilities"][idx] = 1.0
        hps_dictionary["noise_types"] = noise_types

        return hps_dictionary
    else:
        if mode[0] == "vanilla":
            return {
                "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
                "l2": trial.suggest_float("l2", 1e-7, 1e-1, log=True),
                "noise_types": ["none"],
                "noise_probabilities": [0.0],
            }
        elif mode[0] == "batch":
            return {
                "batch_size": trial.suggest_categorical(
                    "batch_size", [32, 64, 128, 256, 512, 1024]
                ),
                "noise_types": ["none"],
                "noise_probabilities": [0.0],
            }
        elif mode[0] == "gradient_gaussian":
            return {
                "noise_gradient_eta": trial.suggest_float(
                    "noise_gradient_eta", 0.0, 1.0
                ),
                "noise_gradient_gamma": trial.suggest_float(
                    "noise_gradient_gamma", 0.0, 1.0
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["gradient_gaussian"],
            }
        elif mode[0] == "target_smoothing":
            return {
                "noise_target_label_smoothing": trial.suggest_float(
                    "noise_target_label_smoothing", 0.0, 0.25
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["target_label_smoothing"],
            }
        elif mode[0] == "model_sp":
            return {
                "noise_model_shrink_and_perturb_mu": trial.suggest_float(
                    "noise_model_shrink_and_perturb_mu", 0.0, 1.0
                ),
                "noise_model_shrink_and_perturb_sigma": trial.suggest_float(
                    "noise_model_shrink_and_perturb_sigma", 1e-7, 0.001, log=True
                ),
                "noise_model_shrink_and_perturb_epoch_frequency": trial.suggest_int(
                    "noise_model_shrink_and_perturb_epoch_frequency", 1, 20
                ),
                "noise_types": ["model_shrink_and_perturb"],
                "noise_probabilities": [1.0],
            }
        elif mode[0] == "input_additive_gaussian":
            return {
                "noise_input_additive_gaussian_sigma": trial.suggest_float(
                    "noise_input_additive_gaussian_sigma", 1e-4, 1e-1, log=True
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["input_additive_gaussian"],
            }
        elif mode[0] == "input_multiplicative_gaussian":
            return {
                "noise_input_multiplicative_gaussian_sigma": trial.suggest_float(
                    "noise_input_multiplicative_gaussian_sigma", 1e-4, 1e-1, log=True
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["input_multiplicative_gaussian"],
            }
        elif mode[0] == "input_additive_uniform":
            return {
                "noise_input_additive_uniform_sigma": trial.suggest_float(
                    "noise_input_additive_uniform_sigma", 1e-4, 1e-1, log=True
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["input_additive_uniform"],
            }
        elif mode[0] == "input_multiplicative_uniform":
            return {
                "noise_input_multiplicative_uniform_sigma": trial.suggest_float(
                    "noise_input_multiplicative_uniform_sigma", 1e-4, 1e-1, log=True
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["input_multiplicative_uniform"],
            }
        elif mode[0] == "input_ods":
            return {
                "noise_input_ods_temperature": trial.suggest_float(
                    "noise_input_ods_temperature", 0.5, 5.0, log=True
                ),
                "noise_input_ods_eta": trial.suggest_float(
                    "noise_input_ods_eta", 1e-4, 1e-1, log=True
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["input_ods"],
            }
        elif mode[0] == "input_augmix":
            return {
                "noise_input_augmix_alpha": trial.suggest_float(
                    "noise_input_augmix_alpha", 0.0, 1.0
                ),
                "noise_input_augmix_severity": trial.suggest_int(
                    "noise_input_augmix_severity", 1, 10
                ),
                "noise_input_augmix_width": trial.suggest_int(
                    "noise_input_augmix_width", 1, 5
                ),
                "noise_input_augmix_chain_depth": trial.suggest_categorical(
                    "noise_input_augmix_chain_depth", [-1, 1, 2, 3]
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["input_augmix"],
            }
        elif mode[0] == "input_target_mixup":
            return {
                "noise_input_target_mixup_alpha": trial.suggest_float(
                    "noise_input_target_mixup_alpha", 0.0, 1.0
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["input_target_mixup"],
            }
        elif mode[0] == "input_target_cmixup":
            return {
                "noise_input_target_cmixup_alpha": trial.suggest_float(
                    "noise_input_target_cmixup_alpha", 0.0, 1.0
                ),
                "noise_input_target_cmixup_sigma": trial.suggest_float(
                    "noise_input_target_cmixup_sigma", 1e-4, 1e2, log=True
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["input_target_cmixup"],
            }
        elif mode[0] == "input_random_crop_horizontal_flip":
            return {
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["input_random_crop_horizontal_flip"],
            }
        elif mode[0] == "activation_additive_gaussian":
            return {
                "noise_activation_additive_gaussian_sigma": trial.suggest_float(
                    "noise_activation_additive_gaussian_sigma", 1e-4, 1e-1, log=True
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["activation_additive_gaussian"],
            }
        elif mode[0] == "activation_multiplicative_gaussian":
            return {
                "noise_activation_multiplicative_gaussian_sigma": trial.suggest_float(
                    "noise_activation_multiplicative_gaussian_sigma",
                    1e-4,
                    1e-1,
                    log=True,
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["activation_multiplicative_gaussian"],
            }
        elif mode[0] == "activation_additive_uniform":
            return {
                "noise_activation_additive_uniform_sigma": trial.suggest_float(
                    "noise_activation_additive_uniform_sigma", 1e-4, 1e-1, log=True
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["activation_additive_uniform"],
            }
        elif mode[0] == "activation_multiplicative_uniform":
            return {
                "noise_activation_multiplicative_uniform_sigma": trial.suggest_float(
                    "noise_activation_multiplicative_uniform_sigma",
                    1e-4,
                    1e-1,
                    log=True,
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["activation_multiplicative_uniform"],
            }
        elif mode[0] == "activation_dropout":
            return {
                "noise_activation_dropout_p": trial.suggest_float(
                    "noise_activation_dropout_p", 0.0, 1.0
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["activation_dropout"],
            }
        elif mode[0] == "weight_additive_gaussian":
            return {
                "noise_weight_additive_gaussian_sigma": trial.suggest_float(
                    "noise_weight_additive_gaussian_sigma", 1e-4, 1e-1, log=True
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["weight_additive_gaussian"],
            }
        elif mode[0] == "weight_multiplicative_gaussian":
            return {
                "noise_weight_multiplicative_gaussian_sigma": trial.suggest_float(
                    "noise_weight_multiplicative_gaussian_sigma",
                    1e-4,
                    1e-1,
                    log=True,
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["weight_multiplicative_gaussian"],
            }
        elif mode[0] == "weight_additive_uniform":
            return {
                "noise_weight_additive_uniform_sigma": trial.suggest_float(
                    "noise_weight_additive_uniform_sigma", 1e-4, 1e-1, log=True
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["weight_additive_uniform"],
            }
        elif mode[0] == "weight_multiplicative_uniform":
            return {
                "noise_weight_multiplicative_uniform_sigma": trial.suggest_float(
                    "noise_weight_multiplicative_uniform_sigma",
                    1e-4,
                    1e-1,
                    log=True,
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["weight_multiplicative_uniform"],
            }
        elif mode[0] == "weight_dropconnect":
            return {
                "noise_weight_dropconnect_p": trial.suggest_float(
                    "noise_weight_dropconnect_p", 0.0, 1.0
                ),
                "noise_probabilities": [
                    trial.suggest_float("noise_probabilities", 0.0, 1.0)
                ],
                "noise_types": ["weight_dropconnect"],
            }
        else:
            raise ValueError("Unknown mode: " + mode)


def objective(
    trial,
    args: argparse.Namespace,
) -> None:
    """Objective function for Optuna optimization."""

    args_trial, _ = utils.parse_args(args)
    current_trial = trial.number
    logging.info("Current trial: " + str(current_trial))

    logging.info("# Starting training #")
    hyperparameters = get_hyperparameters(
        args_trial.dataset, args_trial.architecture, args_trial.hyperparameters
    )
    sampled_hyperparameters = sample_configuration(trial, hyperparameters, args_trial.mode)
    # owerwrite hyperparameters with sampled hyperparameters
    hyperparameters.update(sampled_hyperparameters)

    logging.info("## Loading main model ##")
    model, noises = utils.load_complete_model(args_trial.dataset, args_trial.architecture, hyperparameters, path=None)
    model = utils.model_to_gpu(model, args_trial.gpu)
    logging.info("## Model created: ##")
    logging.info(model.__repr__())

    # Modify training so that it does not do validation
    train(args_trial, model, noises, writer=None, log=None, hyperparameters=hyperparameters, validate=True, trial=trial)

    utils.save_model(model, args_trial.save)

    logging.info("# Finished training #")
    logging.info("# Evaluating with respect to default parameters during training #")

    eval_results, _ = evaluate_model(args_trial, model, hyperparameters)

    # Remove the directory after evaluation
    shutil.rmtree(args_trial.save)

    return [eval_results["nll"]["valid"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("training")

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="define the dataset and thus task",
    )
    parser.add_argument(
        "--mode",
        type=str,
        nargs="+",
        choices=[
            "vanilla",
            "batch",
            "target_smoothing",
            "model_sp",
            "input_additive_gaussian",
            "input_multiplicative_gaussian",
            "input_additive_uniform",
            "input_multiplicative_uniform",
            "input_augmix",
            "input_ods",
            "input_target_mixup",
            "input_target_cmixup",
            "activation_additive_gaussian",
            "activation_multiplicative_gaussian",
            "activation_additive_uniform",
            "activation_multiplicative_uniform",
            "activation_dropout",
            "gradient_gaussian",
            "input_random_crop_horizontal_flip",
            "weight_additive_gaussian",
            "weight_multiplicative_gaussian",
            "weight_additive_uniform",
            "weight_multiplicative_uniform",
            "weight_dropconnect",
        ],
        default="vanilla",
        help="define the dataset and thus task",
    )
    parser.add_argument(
        "--architecture", type=str, default="fc", help="name of the model to use"
    )

    parser.add_argument("--save", type=str, default="./runs", help="experiment directory")
    parser.add_argument("--data_root_dir", type=str, default="~/.torch", help="the default data root directory")
    parser.add_argument("--label", type=str, default="", help="experiment name")

    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--debug", action="store_true", help="whether we are currently debugging"
    )
    parser.add_argument(
        "--mute", action="store_true", help="whether we are currently debugging"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ids")
    parser.add_argument("--hyperparameters", type=json.loads, default={})
    parser.add_argument(
        "--timeout",
        type=int,
        default=86400,
        help="default time for hpo.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="default number of trials for hpo.",
    )
    parser.add_argument(
        "--save_initial_model",
        action="store_true",
        help="whether to save the initial model",
    )
    parser.add_argument(
        "--save_every_epoch",
        action="store_true",
        help="whether to save the model every epoch",
    )
    parser.add_argument(
        "--save_logs",
        action="store_true",
        help="whether to save the training logs",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="whether to save the testing predictions",
    )
    parser.add_argument(
        "--validation_only",
        action="store_true",
        help="whether to only evaluate on the validation set",
    )
    parser.add_argument(
        "--evaluation_augmentations",
        nargs="+",
        default=None,
        help="augmentations to evaluate on",
    )
    parser.add_argument(
        "--evaluation_iterations",
        type=int,
        default=None,
        help="number of batches to evaluate on",
    )

    args, additional_args = parser.parse_known_args()
    
    args.validation_only = True
    args.save_predictions = False

    # Ensure reproducible sampling of configurations
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    start_time = time.time()

    def objective_func(trial):
        return objective(trial, args)

    # Define details of the HPO
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=5, reduction_factor=4, min_early_stopping_rate=0)
    study = optuna.create_study(directions=["minimize"], sampler=sampler, pruner=pruner)

    # Run the HPO
    study.optimize(objective_func, n_trials=args.n_trials, n_jobs=1)
    results = study.trials

    # Process the results
    sampled_configurations = [results[i].params for i in range(len(results))]
    # some values in results can be None, so we need to filter them out and put there large values
    scores_per_configuration = []
    for i in range(len(results)):
        if results[i].values:
            scores_per_configuration.append(results[i].values[0])
        else:
            scores_per_configuration.append(9999.9)
    best_cfg_idx = np.argmin(scores_per_configuration)
    best_cfg = sampled_configurations[best_cfg_idx]
    hpo_time = time.time() - start_time

    summary_dict = {
        "hpo_time": hpo_time,
        "architecture": args.architecture,
        "dataset": args.dataset,
        "num_samples": len(scores_per_configuration),
        "best_hyperparameters": best_cfg,
        "best_score": scores_per_configuration[best_cfg_idx],
        "sampled_configurations": sampled_configurations,
        "scores_per_configuration": scores_per_configuration,
    }

    # Store the results
    hpo_summaries_directory = "hpo_summaries"
    if not os.path.exists(hpo_summaries_directory):
        os.makedirs(hpo_summaries_directory)
    with open(os.path.join(hpo_summaries_directory, args.label + ".json"), "w") as f:
        json.dump(summary_dict, f)
