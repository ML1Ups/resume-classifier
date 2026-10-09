from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def quality_metrics(expected: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
        "f1_macro": float(f1_score(expected, predicted, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(expected, predicted, average="weighted", zero_division=0)),
    }


def create_eda(train: pd.DataFrame, output: Path, preparation: dict | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    counts = train["category"].value_counts().sort_values()
    lengths = train["text"].str.split().str.len()
    height = max(4.5, len(counts) * 0.26 + 1.5)
    fig, ax = plt.subplots(figsize=(11, height))
    counts.plot.barh(ax=ax, color="#4472c4")
    ax.set(title="Training set class distribution", xlabel="Resume count", ylabel="Category")
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=140)
    plt.close(fig)

    groups = list(train.groupby("category", sort=True))
    fig, ax = plt.subplots(figsize=(11, height))
    ax.boxplot(
        [group["text"].str.split().str.len() for _, group in groups],
        tick_labels=[name for name, _ in groups],
        orientation="horizontal",
        showfliers=False,
    )
    ax.set(
        title="Resume length by category (train only; outliers hidden)",
        xlabel="Words",
        ylabel="Category",
    )
    fig.tight_layout()
    fig.savefig(output / "text_lengths.png", dpi=140)
    plt.close(fig)

    vectorizer = CountVectorizer(min_df=1, max_features=20000)
    matrix = vectorizer.fit_transform(train["text"])
    frequencies = np.asarray(matrix.sum(axis=0)).ravel()
    tokens = pd.DataFrame({"token": vectorizer.get_feature_names_out(), "count": frequencies})
    tokens.sort_values("count", ascending=False).head(30).to_csv(
        output / "frequent_tokens.csv",
        index=False,
    )
    summary = {
        "scope": "training split only; validation/test not inspected during EDA",
        "rows": len(train),
        "classes": len(counts),
        "class_counts": counts.to_dict(),
        "word_length": {
            "min": int(lengths.min()),
            "median": float(lengths.median()),
            "max": int(lengths.max()),
        },
        "missing_values": int(train[["text", "category"]].isna().sum().sum()),
        "exact_duplicate_texts": int(train["text"].duplicated().sum()),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "field_coverage": {
            column.removeprefix("_has_"): float(train[column].mean())
            for column in train.columns
            if column.startswith("_has_")
        },
    }
    imbalance = float(counts.max() / counts.min())
    summary["imbalance_ratio"] = imbalance
    coverage = summary["field_coverage"]
    coverage_note = ""
    if coverage:
        coverage_note = (
            f"Hard skills are present in {coverage.get('hard_skills', 0):.1%} of training rows, "
            f"soft skills in {coverage.get('soft_skills', 0):.1%}; many inputs therefore "
            "provide limited evidence about the professional sphere.\n\n"
        )
    (output / "findings.md").write_text(
        "# EDA findings\n\n"
        f"Analyzed {len(train)} training resumes in {len(counts)} categories. "
        f"Largest/smallest class ratio is {imbalance:.2f}. "
        "Macro F1 gives equal weight to each professional category.\n\n"
        f"Median resume length is {lengths.median():.1f} words "
        f"(range {lengths.min()}-{lengths.max()}); sparse TF-IDF is an appropriate "
        "baseline for text classification. Shared vocabulary can confuse adjacent professions, "
        "so word bigrams and character n-grams are compared.\n\n"
        + coverage_note
        + f"There are {summary['exact_duplicate_texts']} exact repeated texts in train. "
        "Same-text/same-label duplicates were removed for the structured export; repeated "
        "texts with different original labels remain together in one group, so they cannot "
        "leak across split boundaries and reflect ambiguity in the source labels.\n\n"
        "No validation or test text was used to fit the vocabulary or choose EDA features. "
        + (
            "The optional demo is synthetic; its scores are not real-world accuracy.\n"
            if (preparation or {}).get("synthetic_demo")
            else "The supplied CV export has uneven class support. Candidate identities and "
            "identical texts are grouped before splitting; targets and IDs are not features. "
            "The two unsupported rare spheres are excluded with an explicit preparation report.\n"
        ),
        encoding="utf-8",
    )
    return summary


def create_diagnostics(frame: pd.DataFrame, predicted: np.ndarray, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    labels = sorted(frame["category"].unique())
    report = classification_report(
        frame["category"],
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(frame["category"], predicted, labels=labels)
    large = len(labels) > 12
    plotted = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1) if large else matrix
    fig, ax = plt.subplots(figsize=(16, 14) if large else (9, 7))
    ConfusionMatrixDisplay(plotted, display_labels=labels).plot(
        ax=ax,
        xticks_rotation=90 if large else 35,
        colorbar=large,
        cmap="Blues",
        include_values=not large,
    )
    ax.set_title("Row-normalized confusion matrix" if large else "Confusion matrix")
    if large:
        ax.tick_params(labelsize=8)
        ax.images[0].set_clim(0, 1)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    rows = frame[["record_id", "category"]].copy()
    rows["prediction"] = predicted
    rows["word_count"] = frame["text"].str.split().str.len()
    rows.to_csv(output / "predictions.csv", index=False)
    rows.loc[rows["category"] != rows["prediction"]].to_csv(output / "errors.csv", index=False)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output / "confusion_matrix.csv")
    confused = [
        (labels[i], labels[j], int(matrix[i, j]))
        for i in range(len(labels))
        for j in range(len(labels))
        if i != j and matrix[i, j] > 0
    ]
    pd.DataFrame(
        sorted(confused, key=lambda item: -item[2]), columns=["expected", "predicted", "count"]
    ).to_csv(
        output / "confused_pairs.csv",
        index=False,
    )
    return report
