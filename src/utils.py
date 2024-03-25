import argparse
import copy
import glob
import logging
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data import get_data_mean_std, get_dataset_options
from src.models import model_factory
from src.noises import noise_factory


def config_logger(save_path: str) -> None:
    """Configure the logger to log info in terminal and file `log.log`."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_path = os.path.join(save_path, "log.log")
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def save_model(model: torch.nn.Module, save_path: str, special_info: str = "") -> None:
    """Save the model to `save_path`."""
    torch.save(model.state_dict(), os.path.join(
        save_path, f"weights{special_info}.pt"))


def save_pickle(data: Any, save_path, overwrite: bool = False) -> None:
    """Save the data to `save_path`."""
    save_path = check_path(save_path) if not overwrite else save_path
    with open(save_path, "wb") as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    """Load the data from `path`."""
    file = open(path, "rb")
    return pickle.load(file)


def load_model(model: torch.nn.Module, path: str) -> None:
    """Load the model from `path`.

    The model is updated in place.

    Args:
        model (torch.nn.Module): The model to load.
        path (str): The path to the state dictiionary.
    """
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    model_dict = model.state_dict()
    pretrained_dict = dict(state_dict.items())
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
    }
    # Perform check if everything is loaded properly
    for key, value in model_dict.items():
        if key not in pretrained_dict:
            raise ValueError(f"Missing key {key} in pretrained model")
        assert (
            value.shape == pretrained_dict[key].shape
        ), f"Shape mismatch for key {key}"
    # Check if there are any extra keys in the pretrained model
    for key, value in pretrained_dict.items():
        if key not in model_dict:
            raise ValueError(f"Extra key {key} in pretrained model")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def load_complete_model(dataset: str,
                        architecture: str,
                        hyperparameters: Dict[str, Any],
                        path: Optional[str] = None) -> Tuple[nn.Module, List[nn.Module]]:
    """Load the complete model from `path`. including the noise layers."""
    model = model_factory(
        dataset=dataset,
        architecture=architecture,
        hyperparameters=hyperparameters,
    )
    # This is needed because the noise can change the model
    noise_kwargs = {
        k: v for k, v in hyperparameters.items() if k.startswith("noise_")
    }
    data_mean, data_std = get_data_mean_std(dataset)
    data_size = get_dataset_options(dataset)[1]
    noise_types = hyperparameters["noise_types"]
    noise_probabilities = hyperparameters["noise_probabilities"]
    noises = noise_factory(
        model=model,
        noises=noise_types,
        noise_probabilities=noise_probabilities,
        noise_kwargs=noise_kwargs,
        data_mean=data_mean,
        data_std=data_std,
        data_size=data_size,
    )
    if path is not None:
        load_model(model, path)
    return model, noises


def create_exp_dir(new_path: str, scripts_path: List[str]) -> str:
    """Create the experiment directory and copy all .py files to it for backup."""
    new_path = check_path(new_path)
    Path(new_path).mkdir(parents=True, exist_ok=True)

    new_scripts_path = os.path.join(new_path, "scripts")
    Path(new_scripts_path).mkdir(parents=True, exist_ok=True)
    for dirpath, _, filenames in os.walk(scripts_path):
        structure = os.path.join(new_scripts_path, dirpath[len(scripts_path):])
        if not os.path.isdir(structure) and "__pycache__" not in dirpath:
            os.mkdir(structure)
        # Now given that we have created that directory, copy all .py files to it
        for file in filenames:
            if file.endswith(".py") and "__pycache__" not in file:
                shutil.copy(os.path.join(dirpath, file), structure)

    # Copy all the .py files also in the current directory and save them under scripts/experiments/
    Path(new_path).mkdir(parents=True, exist_ok=True)
    files = glob.glob("*.py")
    for file in files:
        shutil.copy(file, os.path.join(new_path, "scripts", "experiments"))

    return new_path


def check_path(path: str) -> str:
    """Check if the path exists, if not append a number to it."""
    if os.path.exists(path):
        filename, file_extension = os.path.splitext(path)
        counter = 0
        while os.path.exists(f"{filename}_{counter}{file_extension}"):
            counter += 1
        return f"{filename}_{counter}{file_extension}"
    return path


def model_to_gpu(model: torch.nn.Module, gpu: int) -> torch.nn.Module:
    """Move the model to the GPU."""
    if gpu >= 0:
        device = torch.device(f"cuda:{str(gpu)}")
        model = model.to(device)
    return model


def decompose_experiment_name(experiment_name: str) -> Tuple[str, ...]:
    """Decompose the experiment name into its components."""
    return experiment_name.split("-")


def parse_args(args: argparse.Namespace) -> Tuple[argparse.Namespace, SummaryWriter]:
    """Parse the arguments and create the experiment directory."""
    dataset = args.dataset
    architecture = args.architecture
    save_path = args.save
    new_path = os.path.join(
        save_path,
        "{}-{}-{}".format(dataset, architecture,
                          time.strftime("%Y%m%d-%H%M%S")),
    )
    if args.label != "":
        new_path = os.path.join(
            save_path, "{}-{}".format(new_path, args.label))

    new_path = create_exp_dir(new_path, scripts_path="../calib/")
    # have to create a deep copy of args to avoid repeated paths in save argument
    current_args = copy.deepcopy(args)
    current_args.save = new_path

    if hasattr(current_args, "mute") and current_args.mute:
        sys.stdout = open(os.devnull, "w")

    config_logger(save_path=current_args.save)

    print("Experiment dir : {}".format(current_args.save))
    logging.info("Experiment dir : {}".format(current_args.save))

    writer = SummaryWriter(log_dir=current_args.save + "/", max_queue=5)

    current_args.seed = 0 if not hasattr(
        current_args, "seed") else current_args.seed
    if torch.cuda.is_available() and hasattr(current_args, "gpu") and current_args.gpu != -1:
        logging.info("## GPUs available = {} ##".format(current_args.gpu))
        torch.cuda.set_device(current_args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        torch.cuda.manual_seed(current_args.seed)
    else:
        logging.info("## No GPUs detected ##")
    random.seed(current_args.seed)
    np.random.seed(current_args.seed)
    torch.manual_seed(current_args.seed)
    logging.info("## Args = %s ##", current_args)

    path = os.path.join(current_args.save, "results.pt")
    path = check_path(path)
    results = {}
    save_pickle(results, path, True)
    return current_args, writer
