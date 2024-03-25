from typing import Dict, List, Optional
import sys
import argparse
import numpy as np
import datetime
import logging
import os

sys.path.append("../")

from src.utils import load_pickle, save_pickle, config_logger


def get_dict_path(dictionary: Dict[str, float], path: List[str] = []):
    """A helper function to get the path to the first value in a dictionary that is not a dictionary."""
    for key, value in dictionary.items():
        if type(value) is dict:
            return get_dict_path(dictionary[key], path+[key])
        return path+[key]
    return path


def get_dict_value(dictionary: Dict[str, float], path: List[str] = [], delete: bool = False) -> float:
    """A helper function to get the value of a dictionary at a given path."""
    if len(path) == 1:
        val = dictionary[path[0]]
        if delete:
            dictionary.pop(path[0])
        return val
    else:
        return get_dict_value(dictionary[path[0]], path[1:], delete=delete)


def set_dict_value(dictionary: Dict[str, float], value: float, path: List[str] = []) -> None:
    """A helper function to set the value of a dictionary at a given path."""
    if len(path) == 1:
        dictionary[path[0]] = value
    else:
        if not path[0] in dictionary:
            dictionary[path[0]] = {}
        set_dict_value(dictionary[path[0]], value, path[1:])


def main(folder_paths: List[str], label: str, save_path: Optional[str] = None) -> Dict[str, float]:
    """Averages the results of multiple experiments and saves them in a new folder."""
    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d-%H%M%S")
    if save_path is not None:
        save_path = os.path.join(save_path, label+"-"+time_string)
        os.makedirs(save_path)
        config_logger(save_path=save_path)

    logging.info('# Beginning analysis #')
    final_results = {}
    logging.info('## Loading of result pickles for the experiment ##')

    results = []
    if len(folder_paths) == 1:
        folder_paths = folder_paths[0].split(" ")

    for folder_path in folder_paths:
        result = load_pickle(os.path.join(folder_path, "results.pt"))
        logging.info("### Loading result path: {} ###".format(folder_path))
        logging.info('### Loading result: {} ###'.format(result))

        result_updates = {}
        for key, value in result.items():
            ood_results = []
            for eval_key, eval_value in value.items():
                if '_0' in eval_key or '_1' in eval_key or '_2' in eval_key or '_3' in eval_key or '_4' in eval_key:
                    ood_results.append(eval_value)
            result_updates[key] = np.mean(ood_results)
        for key in result_updates.keys():
            result[key]['ood_test'] = result_updates[key]
        results.append(result)

    assert len(results) > 1

    traversing_result = results[0]
    while len(get_dict_path(traversing_result)) != 0:
        path = get_dict_path(traversing_result)
        values = []
        for result in results:
            try:
                val = get_dict_value(result, path, delete=True)
                if not isinstance(val, dict):
                    values.append(val)
            except Exception as e:
                val = None
                logging.info('### Error: {} ###'.format(e))

        if len(values) == 0 or type(values[0]) == str or len(values) != len(results):
            continue

        values = np.array(values)
        mean = np.nanmean(values)
        std = np.nanstd(values)
        set_dict_value(final_results, (mean, std), path)

    logging.info('## Results: {} ##'.format(final_results))
    if save_path is not None:
        save_pickle(final_results, os.path.join(
            save_path, "results.pt"), overwrite=True)
        logging.info('# Finished #')
        logging.info('## Saved results to {} ##'.format(save_path))

    return final_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser("average_results")
    parser.add_argument('--folder_paths', nargs='+',
                        default='EXP', help='experiment name')
    parser.add_argument('--save_path', type=str, default='EXP',
                        help='path to save the experiment')
    parser.add_argument('--label', type=str, default='',
                        help='label for the experiment')
    main(**vars(parser.parse_args()))