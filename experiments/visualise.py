from typing import Optional, Union, List
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.visualisations.linear_model_interpolation import linear_model_interpolation
from src.visualisations.scalar_metrics import visualise_metrics
from src.visualisations.two_dimensions import landscape_2d
from src.visualisations.two_dimensions_path import landscape_2d_path
from src.training.hyperparameters import get_hyperparameters
from src.utils import decompose_experiment_name, load_pickle
from src.noises import NOISE_MAPPING

LAST_WEIGHTS = "weights_final.pt"
FIRST_WEIGHTS = "weights_initial.pt"

LINEAR_MODEL_INTERPRETATION_FOLDER = "linear_model_interpretation"
SCALAR_METRICS_FOLDER = "scalar_metrics"
LANDSCAPE_2D_FOLDER = "landscape_2d"
LANDSCAPE_2D_LIMIT = 1.0
LANDSCAPE_2D_NUM_POINTS = 10
LANDSCAPE_2D_PATH_FOLDER = "landscape_2d_path"


def visualise_experiment(
    experiment_folder: Union[str, List[str]],
    iterations: Optional[int] = None,
    gpu: int = 0,
    data_root_dir: str = "~/.torch",
    seed: int = 42,
    augmentations: Optional[List[str]] = None,
    debug: bool = False,
    detailed: bool = False,
) -> None:
    """Visualise all the metrics for a given experiment folder.

    The visualisations are stored directly in the experiment folder.
    """
    if isinstance(experiment_folder, str):
        experiment_folder = [experiment_folder]
    for _experiment_folder in experiment_folder:
        # First create a `visualisations` folder in the experiment folder
        visualisations_folder = os.path.join(_experiment_folder, "visualisations")
        os.makedirs(visualisations_folder, exist_ok=True)
        
        print("Visualising experiment folder: {}".format(_experiment_folder))

        # Check if args.pt exists
        if not os.path.exists(os.path.join(_experiment_folder, "args.pt")):
            experiment_name = os.path.basename(os.path.normpath(_experiment_folder))
            fields = decompose_experiment_name(experiment_name)

            dataset = fields[0]
            architecture = fields[1]
            label = "-".join(fields[2:])

            additional_hyperparameters = {
                "batch_size": 1000,
            }
            if debug:
                additional_hyperparameters["batch_size"] = 100

            noise_names = NOISE_MAPPING.keys()
            # Select all the noise types present in the label

            noise_types = []
            noise_probabilities = []
            for noise_name in noise_names:
                if noise_name in label:
                    # The noise hyperparameters are not important since
                    # the noise does not impact the evaluation
                    noise_types.append(noise_name)
                    noise_probabilities.append(0.0)

            additional_hyperparameters["noise_types"] = noise_types
            additional_hyperparameters["noise_probabilities"] = noise_probabilities

            hyperparameters = get_hyperparameters(
                dataset, architecture, additional_hyperparameters
            )
        else:
            args = load_pickle(os.path.join(_experiment_folder, "args.pt"))
            hyperparameters = args.hyperparameters
            dataset = args.dataset
            architecture = args.architecture
            seed = args.seed
            hyperparameters["batch_size"] = 1000
            if debug:
                hyperparameters["batch_size"] = 100

        # First experiment is the linear model interpolation
        linear_model_interpolation_path = os.path.join(
            visualisations_folder, LINEAR_MODEL_INTERPRETATION_FOLDER
        )
        os.makedirs(linear_model_interpolation_path, exist_ok=True)
        
        print("Visualising linear model interpolation")

        linear_model_interpolation(
            model_a_path=os.path.join(_experiment_folder, FIRST_WEIGHTS),
            model_b_path=os.path.join(_experiment_folder, LAST_WEIGHTS),
            save_dir=linear_model_interpolation_path,
            dataset=dataset,
            architecture=architecture,
            iterations=iterations,
            gpu=gpu,
            hyperparameters=hyperparameters,
            data_root_dir=data_root_dir,
            seed=seed,
            augmentations=augmentations,
            debug=debug,
            detailed=detailed,
        )

        # Second observe just the scalar metrics
        
        print("Visualising scalar metrics")
        os.makedirs(
            os.path.join(visualisations_folder, SCALAR_METRICS_FOLDER), exist_ok=True
        )
        visualise_metrics(
            model_dir=_experiment_folder,
            save_dir=os.path.join(visualisations_folder, SCALAR_METRICS_FOLDER),
            labels="",
            dataset=dataset,
            augmentations=augmentations,
        )
        

        # Third observe the 2D landscape
        print("Visualising 2D landscape")
        os.makedirs(
            os.path.join(visualisations_folder, LANDSCAPE_2D_FOLDER), exist_ok=True
        )
        landscape_2d(
            model_path=os.path.join(_experiment_folder, LAST_WEIGHTS),
            save_dir=os.path.join(visualisations_folder, LANDSCAPE_2D_FOLDER),
            dataset=dataset,
            architecture=architecture,
            iterations=iterations,
            gpu=gpu,
            hyperparameters=hyperparameters,
            data_root_dir=data_root_dir,
            seed=seed,
            limit=LANDSCAPE_2D_LIMIT,
            num_points=LANDSCAPE_2D_NUM_POINTS,
            augmentations=augmentations,
            debug=debug,
            detailed=detailed,
        )

        # Forth observe the 2D landscape with a training path
        print("Visualising 2D landscape path")
        os.makedirs(
            os.path.join(visualisations_folder, LANDSCAPE_2D_PATH_FOLDER), exist_ok=True
        )
        landscape_2d_path(
            model_dir=_experiment_folder,
            save_dir=os.path.join(visualisations_folder, LANDSCAPE_2D_PATH_FOLDER),
            dataset=dataset,
            architecture=architecture,
            iterations=iterations,
            gpu=gpu,
            hyperparameters=hyperparameters,
            data_root_dir=data_root_dir,
            seed=seed,
            num_points=LANDSCAPE_2D_NUM_POINTS,
            augmentations=augmentations,
            debug=debug,
            detailed=detailed,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_folder", nargs="+", type=str)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_root_dir", type=str, default="~/.torch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augmentations",
        nargs="+",
        default=None,
        help="augmentations to evaluate on for visualisations",
    )
    parser.add_argument(
        "--debug", action="store_true", help="whether we are currently debugging"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="whether we are currently debugging",
    )

    args = parser.parse_args()
    
    
    # Run the visualisation for all the subfolders of the given experiment folder
    subfolders = [os.path.join(args.experiment_folder[0], f) for f in os.listdir(args.experiment_folder[0]) if os.path.isdir(os.path.join(args.experiment_folder[0], f))]
    visualise_experiment(subfolders, args.iterations, args.gpu, args.data_root_dir, args.seed, args.augmentations, args.debug, args.detailed)

