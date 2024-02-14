from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

global GT_LABELS
GT_LABELS = ["one", "two", "three", "four", "five"]


def get_heatmap_text(cf_matrix: np.ndarray) -> np.ndarray:
    normalized_cf_matrix = cf_matrix / cf_matrix.sum(axis=1)[:, None]
    n = len(cf_matrix)
    text = np.full((n, n), "", dtype=object)
    for y in range(n):
        for x in range(n):
            percentage = normalized_cf_matrix[y, x] * 100
            count = cf_matrix[y, x]
            text[y][x] = f"{percentage:.2f}%\n({count})"
    return text


def export_confusion_matrix(stats: pd.DataFrame, save_path: Path) -> None:
    """
    Exports a confusion matrix to a file.
    :param stats: The stats to use to create the confusion matrix.
    :param save_path: The path to save the confusion matrix to.
    :return: None
    """
    save_path.mkdir(parents=True, exist_ok=True)
    y_true = stats["ground_truth_label"]
    y_pred = stats["predicted_label"]
    cf_matrix = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.heatmap(
        cf_matrix / cf_matrix.sum(axis=1)[:, None],
        annot=get_heatmap_text(cf_matrix),
        fmt="",
        cmap="viridis",
        cbar=False,
        ax=ax,
    )

    # ax.set_title("Confusion Matrix", size=20)
    ax.set_xlabel("Predicted", size=14)
    ax.set_ylabel("Ground Truth", size=14)
    ax.set_xticklabels(GT_LABELS, size=10)
    ax.set_yticklabels(GT_LABELS, size=10)

    cbar = fig.colorbar(
        ax.collections[0], ax=ax, orientation="vertical", fraction=0.15, aspect=12.5
    )
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    # export with transparent background
    plt.savefig(save_path.joinpath("confusion_matrix.png"), dpi=300)
    plt.close()
