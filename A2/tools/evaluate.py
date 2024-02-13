from pathlib import Path
from a2.evaluate.stats import export_confusion_matrix
from argparse import ArgumentParser
import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument("stats", type=str, nargs="+", help="Paths to the stats files.")
    parser.add_argument(
        "--save_path", type=str, help="Path to save the confusion matrix to."
    )
    args = parser.parse_args()
    stats = [pd.read_csv(stat) for stat in args.stats]
    combined_stats = pd.concat(stats, ignore_index=True)
    export_confusion_matrix(combined_stats, Path(args.save_path))


if __name__ == "__main__":
    main()
