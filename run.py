import argparse
import pathlib

from libs.survey_data import SurveyData

import logging

logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Seabed classification.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Dataset directory (leave empty for current dir)",
    )
    parser.add_argument(
        "--inputs", type=str, nargs="*", help="Input areas (JSON files)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Specify features to use (bs: backscatter only, bathy: bathymetr, bpi: BPI)",
        default="bs",
        choices=("bs", "bathy", "bpi"),
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    dataset_path = pathlib.Path(args.dataset_dir) / args.inputs[0]
    SurveyData(dataset_path, dataset_dir=args.dataset_dir, mode=args.mode)


if __name__ == "__main__":
    main()
