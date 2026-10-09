import argparse
import os
import tempfile
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import structlog
from mlflow import MlflowClient
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from resume_classifier import __version__
from resume_classifier.logs import configure_logging
from resume_classifier.ml.artifacts import create_diagnostics, create_eda, quality_metrics
from resume_classifier.ml.dataset import DatasetSplits, lineage, load_and_split

logger = structlog.get_logger(__name__)

BASELINE = "baseline_dummy"
SIMPLICITY = {"logreg_words": 0, "svm_bigrams": 1, "logreg_characters": 2}
EVALUATION_POLICY = {
    "primary_metric": "validation_f1_macro",
    "selection": "max validation macro F1; ties prefer simpler model",
    "test_policy": "evaluate selected model once after selection",
}
INPUT_EXAMPLE = pd.DataFrame({"text": ["Professional experience, skills and education."]})


def candidates(seed: int) -> dict[str, Pipeline]:
    specs = {
        BASELINE: (TfidfVectorizer(), DummyClassifier(strategy="most_frequent")),
        "logreg_words": (
            TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, max_features=20000),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed),
        ),
        "svm_bigrams": (
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=30000),
            LinearSVC(C=0.5, class_weight="balanced", random_state=seed),
        ),
        "logreg_characters": (
            TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=30000
            ),
            LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, random_state=seed),
        ),
    }
    return {
        name: Pipeline(
            [
                ("features", ColumnTransformer([("text", vectorizer, "text")])),
                ("classifier", classifier),
            ]
        )
        for name, (vectorizer, classifier) in specs.items()
    }


def tracked_datasets(splits: DatasetSplits) -> dict[str, PandasDataset]:
    parts = {
        "full": splits.full,
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    return {
        name: mlflow.data.from_pandas(
            frame,
            source=splits.source,
            targets="category",
            name=f"resumes_{name}",
        )
        for name, frame in parts.items()
    }


def log_eda(
    splits: DatasetSplits,
    datasets: dict[str, PandasDataset],
    provenance: dict,
    tags: dict[str, str],
) -> str:
    with (
        tempfile.TemporaryDirectory(prefix="resume-eda-") as directory,
        mlflow.start_run(run_name="eda", tags=tags) as run,
    ):
        mlflow.log_input(datasets["full"], context="source")
        mlflow.log_input(datasets["train"], context="eda")
        summary = create_eda(splits.train, Path(directory), splits.preparation)
        mlflow.log_dict(summary, "eda/summary.json")
        mlflow.log_dict(splits.preparation, "data/preparation.json")
        mlflow.log_dict(provenance, "data/lineage.json")
        mlflow.log_artifacts(directory, artifact_path="eda")
        mlflow.log_params({"seed": splits.seed, "eda_scope": "train_only"})
        return run.info.run_id


def log_diagnostics(frame: pd.DataFrame, predicted: pd.Series, artifact_path: str) -> None:
    with tempfile.TemporaryDirectory(prefix="resume-diagnostics-") as directory:
        report = create_diagnostics(frame, predicted, Path(directory))
        mlflow.log_dict(report, f"{artifact_path}/classification_report.json")
        mlflow.log_artifacts(directory, artifact_path=artifact_path)


def train_candidate(
    name: str,
    model: Pipeline,
    splits: DatasetSplits,
    datasets: dict[str, PandasDataset],
    provenance: dict,
    tags: dict[str, str],
    model_name: str,
) -> dict:
    client = MlflowClient()
    with mlflow.start_run(run_name=name, tags=tags) as run:
        mlflow.log_input(datasets["train"], context="training")
        mlflow.log_input(datasets["validation"], context="validation")
        mlflow.log_dict(provenance, "data/lineage.json")
        mlflow.log_dict(EVALUATION_POLICY, "data/evaluation_policy.json")
        vectorizer = model.named_steps["features"].transformers[0][1]
        classifier = model.named_steps["classifier"]
        mlflow.log_params(
            {
                "seed": splits.seed,
                "model_type": type(classifier).__name__,
                "analyzer": vectorizer.analyzer,
                "ngram_range": str(vectorizer.ngram_range),
                "max_features": vectorizer.max_features or "unlimited",
                "sublinear_tf": vectorizer.sublinear_tf,
                "C": getattr(classifier, "C", "not_applicable"),
                "class_weight": getattr(classifier, "class_weight", "not_applicable"),
                "train_rows": len(splits.train),
                "validation_rows": len(splits.validation),
                "test_rows": len(splits.test),
            }
        )
        started = time.perf_counter()
        model.fit(splits.train[["text"]], splits.train["category"])
        fit_seconds = time.perf_counter() - started
        predicted = model.predict(splits.validation[["text"]])
        metrics = quality_metrics(splits.validation["category"], predicted)
        mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
        mlflow.log_metric("fit_seconds", fit_seconds)
        log_diagnostics(splits.validation, predicted, "validation")
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=infer_signature(INPUT_EXAMPLE, model.predict(INPUT_EXAMPLE)),
            input_example=INPUT_EXAMPLE,
        )
        version = None
        if name != BASELINE:
            version = str(mlflow.register_model(model_info.model_uri, model_name).version)
            client.set_model_version_tag(model_name, version, "candidate", name)
            client.set_model_version_tag(model_name, version, "dataset.sha256", splits.file_sha256)
            client.update_model_version(
                model_name,
                version,
                description=f"{name}; validation macro F1={metrics['f1_macro']:.4f}; "
                f"trained on train split only; seed={splits.seed}",
            )
        return {
            "name": name,
            "run_id": run.info.run_id,
            "version": version,
            "model_uri": model_info.model_uri,
            "fit_seconds": fit_seconds,
            **metrics,
        }


def evaluate_champion(
    best: dict,
    model: Pipeline,
    splits: DatasetSplits,
    datasets: dict[str, PandasDataset],
    results: list[dict],
) -> dict[str, float]:
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_input(datasets["test"], context="testing")
        predicted = model.predict(splits.test[["text"]])
        metrics = quality_metrics(splits.test["category"], predicted)
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        log_diagnostics(splits.test, predicted, "test")
        comparison = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
        comparison["is_champion"] = comparison["name"] == best["name"]
        mlflow.log_text(comparison.to_csv(index=False), "comparison/comparison.csv")
        return metrics


def run_training(
    data: Path,
    tracking_uri: str,
    experiment: str = "resume-classifier-hw3",
    model_name: str = "resume-classifier",
    alias: str = "champion",
    seed: int = 42,
    source: str | None = None,
) -> dict:
    splits = load_and_split(data, seed, source)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    datasets = tracked_datasets(splits)
    provenance = lineage(splits, {name: dataset.digest for name, dataset in datasets.items()})
    tags = {
        "dataset.sha256": splits.file_sha256,
        "dataset.parent_digest": datasets["full"].digest,
        "app.version": __version__,
        "purpose": "homework-3",
    }
    eda_run_id = log_eda(splits, datasets, provenance, tags)
    models = candidates(seed)
    results = [
        train_candidate(
            name,
            model,
            splits,
            datasets,
            provenance,
            {**tags, "eda.run_id": eda_run_id},
            model_name,
        )
        for name, model in models.items()
    ]
    best = min(
        (result for result in results if result["version"]),
        key=lambda result: (-result["f1_macro"], SIMPLICITY[result["name"]]),
    )
    test_metrics = evaluate_champion(best, models[best["name"]], splits, datasets, results)
    MlflowClient().set_registered_model_alias(model_name, alias, best["version"])
    summary = {
        "experiment": experiment,
        "eda_run_id": eda_run_id,
        "tracking_uri": tracking_uri,
        "model_name": model_name,
        "alias": alias,
        "champion": best,
        "test_metrics": test_metrics,
        "runs": results,
        "dataset": {
            "sha256": splits.file_sha256,
            "digest": datasets["full"].digest,
            "source": splits.source,
            "rows": len(splits.full),
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
            "classes": int(splits.full["category"].nunique()),
            "preparation": splits.preparation,
        },
    }
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_dict(summary, "comparison/training_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register resume classification models")
    parser.add_argument("--data", type=Path, default=Path("data/cv_target.csv"))
    parser.add_argument(
        "--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
    )
    parser.add_argument("--experiment", default="resume-classifier-hw3")
    parser.add_argument("--model-name", default="resume-classifier")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", help="Stable original CSV URI for dataset provenance")
    args = parser.parse_args()
    configure_logging("INFO", json_logs=True)
    summary = run_training(
        args.data,
        args.tracking_uri,
        args.experiment,
        args.model_name,
        args.alias,
        args.seed,
        args.source,
    )
    logger.info(
        "training_finished",
        champion=summary["champion"]["name"],
        version=summary["champion"]["version"],
        validation_f1_macro=summary["champion"]["f1_macro"],
        test_f1_macro=summary["test_metrics"]["f1_macro"],
    )


if __name__ == "__main__":
    main()
