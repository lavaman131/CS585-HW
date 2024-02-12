from sklearn.metrics import confusion_matrix
from typing import Sequence
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def export_confusion_matrix(stats: pd.DataFrame, save_path: str) -> None:
    y_true = stats["ground_truth_label"]
    y_pred = stats["predicted_label"]
    cf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(cf_matrix, annot=True, fmt="g")
    plt.savefig(save_path, dpi=300)
