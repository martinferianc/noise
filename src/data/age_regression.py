from typing import Callable, Optional, Tuple

from PIL import Image
import torch
import os
import glob
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
import datetime
import re


class WikiFaceDataset(Dataset):

    md5sum = "f536eb7f5eae229ae8f286184364b42b"
    image_size = (64, 64)

    def __init__(self, root_dir: str,
                 seed: int = 42,
                 test_portion: float = 0.1,
                 transform: Optional[Callable] = None,
                 split: str = 'train') -> None:
        assert split in ['train', 'test']

        self.root_dir = os.path.join(root_dir, "wiki_face")

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            download_and_extract_archive(
                url="https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar",
                download_root=self.root_dir,
                filename="wiki_crop.tar",
                md5=self.md5sum
            )

            paths = glob.glob(os.path.join(
                self.root_dir, "wiki_crop", "*/*.jpg"))

            # Resize the images to the given size
            for path in paths:
                with open(path, "rb") as f:
                    image = Image.open(f)
                    image = image.convert("RGB")
                    image = image.resize(self.image_size)
                    image.save(path)

        paths = glob.glob(os.path.join(self.root_dir, "wiki_crop", "*/*.jpg"))
        self.paths = []
        # Filter out the paths where dob or year of photo taken cannot be parsed
        for path in paths:
            try:
                self._convert_path_to_age(path)
                self.paths.append(path)
            except Exception as e:
                pass

        self.transform = transform
        self.generator = torch.Generator().manual_seed(seed)

        # Split the dataset into train or test
        train_size = int(len(self.paths) * (1 - test_portion))

        indices = torch.randperm(
            len(self.paths), generator=self.generator).tolist()
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        self.indices = train_indices if split == 'train' else test_indices

    def __len__(self) -> int:
        """Return the number of images"""
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the image and the age"""
        path = self.paths[self.indices[index]]
        with open(path, "rb") as f:
            image = Image.open(f)
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        age = self._convert_path_to_age(path)

        return image, age

    def _convert_path_to_age(self, path: str) -> float:
        """Convert the path to age"""
        # The file name is 2786_1921-05-21_1989.jpg
        # The age is 1921-05-21 - 1989-07-01
        # Load in the date of birth
        dob = path.split("/")[-1].split("_")[1]
        # Filter the dob with regex to be YYYY-MM-DD to prevent e.g. 2015-02-16UTC08:04
        dob = re.findall(r"\d{4}-\d{2}-\d{2}", dob)[0]
        # If month or day is 00, set it to 01
        dob = dob.replace("-00", "-01")
        dob = datetime.datetime.strptime(dob, "%Y-%m-%d")
        # Load in the date of photo taken
        photo_year_taken = int(path.split("/")[-1].split("_")[2].split(".")[0])
        # Set the date of photo taken to July 1st
        photo_taken = datetime.datetime(year=photo_year_taken, month=7, day=1)
        # Calculate the age and convert it to years
        age = photo_taken - dob
        assert age.days >= 0, "Age cannot be negative"
        age = age.days / 365.25

        # Normalize the age divide by 100 to prevent the age from being too large
        age /= 100

        return age
