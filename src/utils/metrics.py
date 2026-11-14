import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_metrics(actual, predicted, loss):
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="macro", zero_division=0
    )
    return {
        "loss": loss,
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def format_metrics(prefix, metrics):
    return (
        f"{prefix:<5} | loss {metrics['loss']:.4f} | acc {metrics['accuracy']:.4f} | "
        f"precision {metrics['precision']:.4f} | recall {metrics['recall']:.4f} | f1 {metrics['f1']:.4f}"
    )


def to_wandb(prefix, metrics):
    return {
        f"{prefix}/Loss": metrics["loss"],
        f"{prefix}/Accuracy": metrics["accuracy"],
        f"{prefix}/Precision": metrics["precision"],
        f"{prefix}/Recall": metrics["recall"],
        f"{prefix}/F1-Score": metrics["f1"],
    }


def save_confusion_matrix(actual, predicted, class_names, title, save_path):
    cm = confusion_matrix(actual, predicted, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        cm,
        annot=False,  # 60x60 numbers are unreadable, colors tell the story
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Prediction", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title(title, fontsize=17, pad=20)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
