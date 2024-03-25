import logging
import os
from typing import List, Optional, Tuple

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split

from src.data.ng import NewsGroupDataset
from src.data.sst import SSTDataset
from src.data.transformations import RotationDataset
from src.data.uci import UCI
from src.data.age_regression import WikiFaceDataset
from src.third_party.corruptions import VisionCorruption, TabularCorruption
from src.third_party.tiny import TinyImageNet

SVHN_MEAN, SVHN_STD = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
TINYIMAGENET_MEAN, TINYIMAGENET_STD = (
    0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
WIKIFACE_MEAN, WIKIFACE_STD = (
    0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
ROTATIONS = [0, 90]

DATA_MEAN = {
    "cifar10": CIFAR_MEAN,
    "rotated_cifar10": CIFAR_MEAN,
    "cifar100": CIFAR_MEAN,
    "rotated_cifar100": CIFAR_MEAN,
    "svhn": SVHN_MEAN,
    "rotated_svhn": SVHN_MEAN,
    "tinyimagenet": TINYIMAGENET_MEAN,
    "rotated_tinyimagenet": TINYIMAGENET_MEAN,
    "wiki_face": WIKIFACE_MEAN,
    "regression_concrete": None,
    "regression_energy": None,
    "regression_boston": None,
    "regression_wine": None,
    "regression_yacht": None,
    "classification_wine": None,
    "classification_toxicity": None,
    "classification_abalone": None,
    "classification_students": None,
    "classification_adult": None,
    "newsgroup": None,
    "sst": None,
}
DATA_STD = {
    "cifar10": CIFAR_STD,
    "rotated_cifar10": CIFAR_STD,
    "cifar100": CIFAR_STD,
    "rotated_cifar100": CIFAR_STD,
    "svhn": SVHN_STD,
    "rotated_svhn": SVHN_STD,
    "tinyimagenet": TINYIMAGENET_STD,
    "rotated_tinyimagenet": TINYIMAGENET_STD,
    "wiki_face": WIKIFACE_STD,
    "regression_concrete": None,
    "regression_energy": None,
    "regression_boston": None,
    "regression_wine": None,
    "regression_yacht": None,
    "classification_wine": None,
    "classification_toxicity": None,
    "classification_abalone": None,
    "classification_students": None,
    "classification_adult": None,
    "newsgroup": None,
    "sst": None,
}

DATA_SIZE = {
    "cifar10": (3, 32, 32),
    "rotated_cifar10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "rotated_cifar100": (3, 32, 32),
    "svhn": (3, 32, 32),
    "rotated_svhn": (3, 32, 32),
    "tinyimagenet": (3, 64, 64),
    "rotated_tinyimagenet": (3, 64, 64),
    "wiki_face": (3, 64, 64),
    "regression_concrete": (8),
    "regression_energy": (9),
    "regression_boston": (13),
    "regression_wine": (11),
    "regression_yacht": (6),
    "classification_wine": (11),
    "classification_toxicity": (1203),
    "classification_abalone": (8),
    "classification_students": (32),
    "classification_adult": (14),
    "newsgroup": (100, 100),
    "sst": (50, 100),
}
OUTPUT_SIZE = {
    "cifar10": 10,
    "rotated_cifar10": 2,
    "cifar100": 100,
    "rotated_cifar100": 2,
    "svhn": 10,
    "rotated_svhn": 2,
    "tinyimagenet": 200,
    "rotated_tinyimagenet": 2,
    "wiki_face": 2,
    "regression_concrete": 2,
    "regression_energy": 2,
    "regression_boston": 2,
    "regression_wine": 2,
    "regression_yacht": 2,
    "classification_wine": 6,
    "classification_toxicity": 3,
    "classification_abalone": 3,
    "classification_students": 18,
    "classification_adult": 2,
    "newsgroup": 20,
    "sst": 2,
}
TASKS = {
    "cifar10": "classification",
    "rotated_cifar10": "regression",
    "cifar100": "classification",
    "rotated_cifar100": "regression",
    "svhn": "classification",
    "rotated_svhn": "regression",
    "tinyimagenet": "classification",
    "rotated_tinyimagenet": "regression",
    "wiki_face": "regression",
    "regression_concrete": "regression",
    "regression_energy": "regression",
    "regression_boston": "regression",
    "regression_wine": "regression",
    "regression_yacht": "regression",
    "classification_wine": "classification",
    "classification_toxicity": "classification",
    "classification_abalone": "classification",
    "classification_students": "classification",
    "classification_adult": "classification",
    "newsgroup": "classification",
    "sst": "classification",
}

AUGMENTATIONS = {
    "cifar10": VisionCorruption.corruption_names,
    "rotated_cifar10": VisionCorruption.corruption_names,
    "cifar100": VisionCorruption.corruption_names,
    "rotated_cifar100": VisionCorruption.corruption_names,
    "svhn": VisionCorruption.corruption_names,
    "rotated_svhn": VisionCorruption.corruption_names,
    "tinyimagenet": VisionCorruption.corruption_names,
    "rotated_tinyimagenet": VisionCorruption.corruption_names,
    "wiki_face": VisionCorruption.corruption_names,
    "regression_concrete": TabularCorruption.corruption_names,
    "regression_energy": TabularCorruption.corruption_names,
    "regression_boston": TabularCorruption.corruption_names,
    "regression_wine": TabularCorruption.corruption_names,
    "regression_yacht": TabularCorruption.corruption_names,
    "classification_wine": TabularCorruption.corruption_names,
    "classification_toxicity": TabularCorruption.corruption_names,
    "classification_abalone": TabularCorruption.corruption_names,
    "classification_students": TabularCorruption.corruption_names,
    "classification_adult": TabularCorruption.corruption_names,
    "newsgroup": [],
    "sst": [],
}

AUGMENTATIONS_SCALES = {
    "cifar10": None,
    "rotated_cifar10": None,
    "cifar100": None,
    "rotated_cifar100": None,
    "svhn": None,
    "rotated_svhn": None,
    "tinyimagenet": None,
    "rotated_tinyimagenet": None,
    "wiki_face": None,
    "regression_concrete": 0.0206913808111479,
    "regression_energy": 0.05455594781168514,
    "regression_boston": 0.08858667904100823,
    "regression_wine": 0.004832930238571752,
    "regression_yacht": 0.08858667904100823,
    "classification_wine": 0.007847599703514606,
    "classification_toxicity": 0.3792690190732246,
    "classification_abalone": 0.08858667904100823,
    "classification_students": 0.23357214690901212,
    "classification_adult": 0.14384498882876628,
    "newsgroup": None,
    "sst": None,
}

AUGMENTATION_LEVELS = {
    "cifar10": len(VisionCorruption.corruption_names)
    * [range(VisionCorruption.levels)],
    "rotated_cifar10": len(VisionCorruption.corruption_names)
    * [range(VisionCorruption.levels)],
    "cifar100": len(VisionCorruption.corruption_names)
    * [range(VisionCorruption.levels)],
    "rotated_cifar100": len(VisionCorruption.corruption_names)
    * [range(VisionCorruption.levels)],
    "svhn": len(VisionCorruption.corruption_names) * [range(VisionCorruption.levels)],
    "rotated_svhn": len(VisionCorruption.corruption_names)
    * [range(VisionCorruption.levels)],
    "tinyimagenet": len(VisionCorruption.corruption_names)
    * [range(VisionCorruption.levels)],
    "rotated_tinyimagenet": len(VisionCorruption.corruption_names)
    * [range(VisionCorruption.levels)],
    "wiki_face": len(VisionCorruption.corruption_names)
    * [range(VisionCorruption.levels)],
    "regression_concrete": len(TabularCorruption.corruption_names)
    * [range(TabularCorruption.levels)],
    "regression_energy": len(TabularCorruption.corruption_names)
    * [range(TabularCorruption.levels)],
    "regression_boston": len(TabularCorruption.corruption_names)
    * [range(TabularCorruption.levels)],
    "regression_wine": len(TabularCorruption.corruption_names)
    * [range(TabularCorruption.levels)],
    "regression_yacht": len(TabularCorruption.corruption_names)
    * [range(TabularCorruption.levels)],
    "classification_wine": len(TabularCorruption.corruption_names)
    * [range(TabularCorruption.levels)],
    "classification_toxicity": len(TabularCorruption.corruption_names)
    * [range(TabularCorruption.levels)],
    "classification_abalone": len(TabularCorruption.corruption_names)
    * [range(TabularCorruption.levels)],
    "classification_students": len(TabularCorruption.corruption_names)
    * [range(TabularCorruption.levels)],
    "classification_adult": len(TabularCorruption.corruption_names)
    * [range(TabularCorruption.levels)],
    "newsgroup": [],
    "sst": [],
}

CLASSIFICATION_DATASETS = [
    "cifar10",
    "cifar100",
    "svhn",
    "tinyimagenet",
    "newsgroup",
    "sst",
    "classification_wine",
    "classification_toxicity",
    "classification_abalone",
    "classification_students",
    "classification_adult",
]
REGRESSION_DATASETS = [
    "rotated_cifar10",
    "rotated_cifar100",
    "rotated_svhn",
    "rotated_tinyimagenet",
    "wiki_face",
    "regression_concrete",
    "regression_energy",
    "regression_boston",
    "regression_wine",
    "regression_yacht",
]
DATASETS = CLASSIFICATION_DATASETS + REGRESSION_DATASETS


def get_dataloaders(
    dataset: str,
    batch_size: int,
    seed: int,
    level: Optional[int] = None,
    augmentation: str = "",
    valid_portion: float = 0.1,
    test_portion: float = 0.2,
    data_root_dir: str = "~/.torch/",
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """Get train, valid, and test loaders for a dataset.

    Args:
        dataset (str): Name of the dataset.
        batch_size (int): Batch size.
        seed (int): Seed for the random number generator.
        level (int): Level of augmentation.
        augmentation (str): Type of augmentation.
        valid_portion (float): Portion of the training set to use for validation.
        test_portion (float): Portion of the training set to use for testing.
        data_root_dir (str): Root directory for the dataset.
    """
    if dataset not in DATASETS:
        raise ValueError("Unknown dataset: {}".format(dataset))

    if augmentation not in VisionCorruption.corruption_names + [""] + TabularCorruption.corruption_names:
        raise ValueError("Unknown augmentation: {}".format(augmentation))

    data_root_dir = os.path.expanduser(data_root_dir)

    if dataset in [
        "cifar10",
        "cifar100",
        "svhn",
        "tinyimagenet",
        "rotated_cifar10",
        "rotated_cifar100",
        "rotated_svhn",
        "rotated_tinyimagenet",
        "wiki_face",
    ]:
        test_transform = []
        if augmentation != "":
            test_transform.append(
                VisionCorruption(
                    corruption_name=augmentation,
                    severity=level,
                    img_size=DATA_SIZE[dataset],
                )
            )
        test_transform.append(transforms.ToTensor())

        test_transform.append(
            transforms.Normalize(DATA_MEAN[dataset], DATA_STD[dataset])
        )
        test_transform = transforms.Compose(test_transform)

        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(DATA_MEAN[dataset], DATA_STD[dataset]),
            ]
        )
    elif dataset in [
        "regression_concrete",
        "regression_energy",
        "regression_boston",
        "regression_wine",
        "regression_yacht",
    ] or dataset in [
        "classification_wine",
        "classification_toxicity",
        "classification_abalone",
        "classification_students",
        "classification_adult",
    ]:
        test_transform = []
        if augmentation != "":
            test_transform.append(
                TabularCorruption(
                    corruption_name=augmentation,
                    severity=level,
                    dataset_scale=AUGMENTATIONS_SCALES[dataset],
                )
            )
        test_transform = transforms.Compose(test_transform)
        train_transform = transforms.Compose([])
    elif dataset == "newsgroup" or dataset == "sst":
        # There is no augmentation for NLP datasets
        pass
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    train_data: torch.utils.data.Dataset = None
    test_data: torch.utils.data.Dataset = None
    if dataset == "cifar10":
        train_data = datasets.CIFAR10(
            root=data_root_dir, train=True, download=True, transform=train_transform
        )
        test_data = datasets.CIFAR10(
            root=data_root_dir, train=False, download=True, transform=test_transform
        )
    elif dataset == "rotated_cifar10":
        # The transform is applied in the wrapper to guarantee that rotation is applied first!
        train_data = RotationDataset(
            dataset=datasets.CIFAR10(
                root=data_root_dir, train=True, download=True),
            transform=train_transform,
            rotation=ROTATIONS,
            seed=seed,
        )
        test_data = RotationDataset(
            dataset=datasets.CIFAR10(
                root=data_root_dir, train=False, download=True),
            transform=test_transform,
            rotation=ROTATIONS,
            seed=seed,
        )
    elif dataset == "cifar100":
        train_data = datasets.CIFAR100(
            root=data_root_dir, train=True, download=True, transform=train_transform
        )
        test_data = datasets.CIFAR100(
            root=data_root_dir, train=False, download=True, transform=test_transform
        )
    elif dataset == "rotated_cifar100":
        # The transform is applied in the wrapper to guarantee that rotation is applied first!
        train_data = RotationDataset(
            dataset=datasets.CIFAR100(
                root=data_root_dir, train=True, download=True),
            transform=train_transform,
            rotation=ROTATIONS,
            seed=seed,
        )
        test_data = RotationDataset(
            dataset=datasets.CIFAR100(
                root=data_root_dir, train=False, download=True),
            transform=test_transform,
            rotation=ROTATIONS,
            seed=seed,
        )
    elif dataset == "svhn":
        train_data = datasets.SVHN(
            root=data_root_dir, split="train", download=True, transform=train_transform
        )
        test_data = datasets.SVHN(
            root=data_root_dir, split="test", download=True, transform=test_transform
        )
    elif dataset == "rotated_svhn":
        # The transform is applied in the wrapper to guarantee that rotation is applied first!
        train_data = RotationDataset(
            dataset=datasets.SVHN(root=data_root_dir,
                                  split="train", download=True),
            transform=train_transform,
            rotation=ROTATIONS,
            seed=seed,
        )
        test_data = RotationDataset(
            dataset=datasets.SVHN(root=data_root_dir,
                                  split="test", download=True),
            transform=test_transform,
            rotation=ROTATIONS,
            seed=seed,
        )
    elif dataset == "tinyimagenet":
        train_data = TinyImageNet(
            root=data_root_dir, split="train", download=True, transform=train_transform
        )
        test_data = TinyImageNet(
            root=data_root_dir, split="val", download=True, transform=test_transform
        )
    elif dataset == "rotated_tinyimagenet":
        train_data = RotationDataset(
            dataset=TinyImageNet(root=data_root_dir,
                                 split="train", download=True),
            transform=train_transform,
            rotation=ROTATIONS,
            seed=seed,
        )
        test_data = RotationDataset(
            dataset=TinyImageNet(root=data_root_dir,
                                 split="val", download=True),
            transform=test_transform,
            rotation=ROTATIONS,
            seed=seed,
        )
    elif dataset == "wiki_face":
        train_data = WikiFaceDataset(
            root_dir=data_root_dir,
            split="train",
            transform=train_transform,
            seed=seed,
            test_portion=test_portion,
        )
        test_data = WikiFaceDataset(
            root_dir=data_root_dir,
            split="test",
            transform=test_transform,
            seed=seed,
            test_portion=test_portion,
        )
    elif dataset in [
        "regression_concrete",
        "regression_energy",
        "regression_boston",
        "regression_wine",
        "regression_yacht",
        "classification_wine",
        "classification_toxicity",
        "classification_abalone",
        "classification_students",
        "classification_adult",
    ]:
        data = dataset.split("_")[1]
        task = dataset.split("_")[0]
        train_data = UCI(
            root=data_root_dir,
            dataset=data,
            train=True,
            test_portion=test_portion,
            seed=seed,
            task=task,
            transform=train_transform,
        )
        test_data = UCI(
            root=data_root_dir,
            dataset=data,
            train=False,
            test_portion=test_portion,
            seed=seed,
            task=task,
            transform=test_transform,
        )
    elif dataset == "newsgroup":
        train_data = NewsGroupDataset(root=data_root_dir, train=True)
        test_data = NewsGroupDataset(root=data_root_dir, train=False)
    elif dataset == "sst":
        train_data = SSTDataset(root=data_root_dir, train=True)
        test_data = SSTDataset(root=data_root_dir, train=False)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    valid_split = int(len(train_data) * valid_portion)
    train_split = len(train_data) - valid_split
    train_data, valid_data = random_split(
        train_data,
        [train_split, valid_split],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    logging.info("Train: {}".format(len(train_data)))
    logging.info("Valid: {}".format(len(valid_data)))
    logging.info("Test: {}".format(len(test_data)))
    return train_loader, valid_loader, test_loader


def get_dataset_options(
    name: str,
) -> Tuple[str, Tuple[int, int, int], int, List[List[int]], List[str]]:
    """Get the options for a dataset.

    Args:
        name (str): Name of the dataset.

    Returns:
        task (str): Task of the dataset.
        input_shape (tuple): Shape of the input.
        output_shape (int): Number of outputs.
        levels (list): Levels of the dataset augmentations.
        augmentations (list): List of augmentations.
    """
    if name not in DATASETS:
        raise ValueError("Unknown dataset: {}".format(name))

    input_shape = DATA_SIZE[name]
    num_outputs = OUTPUT_SIZE[name]
    task = TASKS[name]
    levels = AUGMENTATION_LEVELS[name]
    augmentations = AUGMENTATIONS[name]
    return task, input_shape, num_outputs, levels, augmentations


def get_data_mean_std(
    name: str,
) -> Tuple[Optional[Tuple[float, ...]], Optional[Tuple[float, ...]]]:
    """Get the mean and standard deviation of a dataset.

    Args:
        name (str): Name of the dataset.

    Returns:
        mean (tuple): Mean of the dataset.
        std (tuple): Standard deviation of the dataset.
    """
    return DATA_MEAN[name], DATA_STD[name]
