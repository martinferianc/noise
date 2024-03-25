from typing import Callable

import os
import shutil
import urllib.request as request
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder

from src.data.transformations import Normalise


class UCI:
    """Data module for the UCI datasets.

    Args:
        dataset (str): Name of the dataset to use.
        root (str): Root directory where the dataset should be stored.
        train (bool): If true, returns the training dataset, otherwise returns the test dataset.
        test_portion (float): Portion of the training data to use for testing.
        seed (int): Random seed to use.
        task (str): Task to perform. One of the following: regression, classification.
        transform (Callable): A function/transform that takes in a datapoint and returns a transformed version.
    """

    def __init__(
        self,
        dataset: str,
        root: str,
        train: bool = True,
        test_portion: float = 0.2,
        seed: int = 0,
        task: str = "regression",
        transform: Callable = None,
    ) -> None:
        assert task in ["regression",
                        "classification"], f"Task {task} not supported"
        assert dataset in [
            "concrete",
            "energy",
            "boston",
            "wine",
            "yacht",
            "toxicity",
            "abalone",
            "students",
            "adult",
        ], f"Dataset {dataset} not supported"
        self._dataset = dataset
        self._root = os.path.join(os.path.expanduser(root), "uci", dataset)
        self._test_portion = test_portion
        self._train = train
        self._seed = seed
        self._task = task

        self._data_mean: torch.Tensor = None
        self._data_std: torch.Tensor = None
        self._data: TensorDataset = None

        data = self._download_data()
        train_dataset, test_dataset = self._split_train_test(data)

        # The mean and standard deviation of the features.
        self._data_mean, self._data_std = self._mean_std(train_dataset, 0)
        self._data_normalizer = Normalise(self._data_mean, self._data_std)
        self._transform = transform

        if task == "regression":
            # The mean and standard deviation of the targets.
            self._target_mean, self._target_std = self._mean_std(
                train_dataset, 1)
            self._target_normalizer = Normalise(
                self._target_mean, self._target_std)
        else:
            self._target_mean, self._target_std = None, None
            self._target_normalizer = None

        if self._train:
            self._data = train_dataset
        else:
            self._data = test_dataset

    def _split_train_test(
        self, dataset: TensorDataset
    ) -> Tuple[TensorDataset, TensorDataset]:
        """Splits the dataset into training and testing sets."""
        train_size = int((1.0 - self._test_portion) * len(dataset))
        test_size = len(dataset) - train_size
        return random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(self._seed),
        )

    def _mean_std(
        self, dataset: TensorDataset, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A helper function to compute the mean and standard deviation for `index`th column of the dataset."""
        mean = torch.mean(
            torch.stack([dataset[i][index] for i in range(len(dataset))]), dim=0
        )
        std = torch.std(
            torch.stack([dataset[i][index] for i in range(len(dataset))]), dim=0
        )
        return mean, std

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the `index`th data point."""
        data, target = self._data[index]
        # No transformation for targets.
        if self._transform is not None:
            data = self._transform(data)
        data = self._data_normalizer(data)
        target = (
            self._target_normalizer(target)
            if self._target_normalizer is not None
            else target
        )
        return data, target

    def __len__(self) -> int:
        """Returns the size of the dataset."""
        return len(self._data)

    def _download_data(self) -> TensorDataset:
        """Downloads the dataset."""
        data: pd.DataFrame = None
        # Create the root directory if it does not exist.
        if not os.path.exists(self._root):
            os.makedirs(self._root, exist_ok=True)
        if self._dataset == "concrete":
            if not os.path.exists(os.path.join(self._root, "Concrete_Data.xls")):
                request.urlretrieve(
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
                    "Concrete_Data.xls",
                )
                shutil.move(
                    "Concrete_Data.xls", os.path.join(
                        self._root, "Concrete_Data.xls")
                )
            data = pd.read_excel(os.path.join(self._root, "Concrete_Data.xls"))
        elif self._dataset == "energy":
            if not os.path.exists(os.path.join(self._root, "ENB2012_data.xlsx")):
                request.urlretrieve(
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
                    "ENB2012_data.xlsx",
                )
                shutil.move(
                    "ENB2012_data.xlsx", os.path.join(
                        self._root, "ENB2012_data.xlsx")
                )
            data = pd.read_excel(os.path.join(self._root, "ENB2012_data.xlsx"))
        elif self._dataset == "boston":
            if not os.path.exists(os.path.join(self._root, "housing.data")):
                request.urlretrieve(
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                    "housing.data",
                )
                shutil.move("housing.data", os.path.join(
                    self._root, "housing.data"))
            data = pd.read_csv(
                os.path.join(self._root, "housing.data"),
                delim_whitespace=True,
                header=None,
            )
        elif self._dataset == "wine":
            if not os.path.exists(os.path.join(self._root, "winequality-red.csv")):
                request.urlretrieve(
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                    "winequality-red.csv",
                )
                shutil.move(
                    "winequality-red.csv",
                    os.path.join(self._root, "winequality-red.csv"),
                )
            data = pd.read_csv(os.path.join(
                self._root, "winequality-red.csv"), sep=";")
        elif self._dataset == "yacht":
            if not os.path.exists(os.path.join(self._root, "yacht_hydrodynamics.data")):
                request.urlretrieve(
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
                    "yacht_hydrodynamics.data",
                )
                shutil.move(
                    "yacht_hydrodynamics.data",
                    os.path.join(self._root, "yacht_hydrodynamics.data"),
                )
            data = pd.read_csv(
                os.path.join(self._root, "yacht_hydrodynamics.data"),
                delim_whitespace=True,
                header=None,
            )
        elif self._dataset == "toxicity":
            if not os.path.exists(os.path.join(self._root, "toxicity-2.zip")):
                request.urlretrieve(
                    "https://archive.ics.uci.edu/static/public/728/toxicity-2.zip",
                    "toxicity-2.zip",
                )
                # Unzip into its own directory.
                shutil.move(
                    "toxicity-2.zip", os.path.join(self._root,
                                                   "toxicity-2.zip")
                )
                shutil.unpack_archive(
                    os.path.join(self._root, "toxicity-2.zip"),
                    os.path.join(self._root, "toxicity-2"),
                )
            data = pd.read_csv(
                os.path.join(self._root, "toxicity-2", "data.csv"), header=1
            ).dropna()

        elif self._dataset == "abalone":
            if not os.path.exists(os.path.join(self._root, "abalone.data")):
                request.urlretrieve(
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
                    "abalone.data",
                )
                shutil.move("abalone.data", os.path.join(
                    self._root, "abalone.data"))
            data = pd.read_csv(
                os.path.join(self._root, "abalone.data"), header=None
            ).dropna()

        elif self._dataset == "students":
            if not os.path.exists(os.path.join(self._root, "student", "student-mat.csv")):
                request.urlretrieve(
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
                    "student.zip",
                )
                shutil.move("student.zip", os.path.join(
                    self._root, "student.zip"))
                shutil.unpack_archive(
                    os.path.join(self._root, "student.zip"),
                    os.path.join(self._root, "student"),
                )

            data = pd.read_csv(
                os.path.join(self._root, "student", "student-mat.csv"), sep=";"
            ).dropna()

        elif self._dataset == "adult":
            if not os.path.exists(os.path.join(self._root, "adult", "adult.data")):
                request.urlretrieve(
                    "https://archive.ics.uci.edu/static/public/2/adult.zip",
                    "adult.zip",
                )
                shutil.move("adult.zip", os.path.join(self._root, "adult.zip"))
                shutil.unpack_archive(
                    os.path.join(self._root, "adult.zip"),
                    os.path.join(self._root, "adult"),
                )

            data = pd.read_csv(
                os.path.join(self._root, "adult", "adult.data"), header=None
            ).dropna()

        data = data.dropna().to_numpy()
        # Convert any non-numerical features to numerical features.
        for i in range(data.shape[1]):
            try:
                data[:, i] = data[:, i].astype(float)
            except:
                data[:, i] = LabelEncoder().fit_transform(data[:, i])

        if self._task == "regression":
            if self._dataset == "concrete":
                features = range(8)
                targets = [8]
            elif self._dataset == "wine":
                # Predict the alcohol content.
                features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
                targets = [10]
            elif self._dataset == "energy":
                features = range(9)
                targets = [9]
            elif self._dataset == "boston":
                features = range(13)
                targets = [13]
            elif self._dataset == "yacht":
                features = range(6)
                targets = [6]
            else:
                raise NotImplementedError(
                    f"Dataset {self._dataset} not supported")
            inputs, targets = data[:, features], data[:, targets]
            inputs, targets = (
                torch.from_numpy(inputs).float(),
                torch.from_numpy(targets).float().squeeze(),
            )
        else:
            if self._dataset == "wine":
                features = range(11)
                targets = [11]
            elif self._dataset == "toxicity":
                features = range(1203)
                targets = [1203]
            elif self._dataset == "abalone":
                features = range(1, 9)
                targets = [0]
            elif self._dataset == "students":
                features = range(32)
                targets = [32]
            elif self._dataset == "adult":
                features = range(14)
                targets = [14]
            else:
                raise NotImplementedError(
                    f"Dataset {self._dataset} not supported")
            inputs, targets = data[:, features], data[:, targets]
            le = LabelEncoder()
            targets = le.fit_transform(targets)
            inputs = inputs.astype(float)
            inputs, targets = (
                torch.from_numpy(inputs).float(),
                torch.from_numpy(targets).long().squeeze(),
            )
        return TensorDataset(inputs, targets)
