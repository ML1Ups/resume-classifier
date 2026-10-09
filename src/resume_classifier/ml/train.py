"""python -m resume_classifier.ml.train --data data/cv_target.csv"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from resume_classifier import __version__
from resume_classifier.ml.artifacts import create_diagnostics, create_eda, quality_metrics
from resume_classifier.ml.dataset import DatasetSplits, lineage, load_and_split


def candidates(seed: int) -> dict[str, Pipeline]:
    specs = {
        "baseline_dummy": (TfidfVectorizer(), DummyClassifier(strategy="most_frequent")),
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


def tracked_datasets(splits: DatasetSplits) -> dict:
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


def run_training(
    data: Path,
    tracking_uri: str,
    experiment: str = "resume-classifier-hw3",
    model_name: str = "resume-classifier",
    alias: str = "champion",
    seed: int = 42,
    output: Path = Path("results"),
    source: str | None = None,
) -> dict:
    splits = load_and_split(data, seed, source)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    client = MlflowClient(tracking_uri=tracking_uri)
    datasets = tracked_datasets(splits)
    provenance = lineage(splits, {name: ds.digest for name, ds in datasets.items()})
    output.mkdir(parents=True, exist_ok=True)
    base_tags = {
        "dataset.sha256": splits.file_sha256,
        "dataset.parent_digest": datasets["full"].digest,
        "app.version": __version__,
        "purpose": "homework-3",
    }

    with (
        tempfile.TemporaryDirectory(prefix="resume-eda-") as tmp,
        mlflow.start_run(run_name="eda", tags=base_tags) as eda_run,
    ):
        mlflow.log_input(datasets["full"], context="source")
        mlflow.log_input(datasets["train"], context="eda")
        summary = create_eda(splits.train, Path(tmp), splits.preparation)
        mlflow.log_dict(summary, "eda/summary.json")
        mlflow.log_dict(splits.preparation, "data/preparation.json")
        mlflow.log_dict(provenance, "data/lineage.json")
        mlflow.log_artifacts(tmp, artifact_path="eda")
        mlflow.log_params({"seed": seed, "eda_scope": "train_only"})
        eda_run_id = eda_run.info.run_id

    results = []
    fitted = {}
    for name, model in candidates(seed).items():
        with mlflow.start_run(run_name=name, tags=base_tags) as run:
            mlflow.set_tag("eda.run_id", eda_run_id)
            mlflow.log_input(datasets["train"], context="training")
            mlflow.log_input(datasets["validation"], context="validation")
            mlflow.log_dict(provenance, "data/lineage.json")
            mlflow.log_dict(
                {
                    "primary_metric": "validation_f1_macro",
                    "selection": "max validation macro F1; ties prefer simpler model",
                    "test_policy": "evaluate selected model once after selection",
                },
                "data/evaluation_policy.json",
            )
            vectorizer = model.named_steps["features"].transformers[0][1]
            classifier = model.named_steps["classifier"]
            mlflow.log_params(
                {
                    "seed": seed,
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
            with tempfile.TemporaryDirectory(prefix="resume-diagnostics-") as tmp:
                report = create_diagnostics(splits.validation, predicted, Path(tmp))
                mlflow.log_dict(report, "validation/classification_report.json")
                mlflow.log_artifacts(tmp, artifact_path="validation")
            # Persist a generic schema example, rather than real personal resume text.
            example = pd.DataFrame({"text": ["Professional experience, skills and education."]})
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=infer_signature(example, model.predict(example)),
                input_example=example,
                pip_requirements=[
                    "mlflow-skinny==3.10.0",
                    "scikit-learn==1.7.2",
                    "pandas==2.3.3",
                    "numpy==2.2.6",
                ],
            )
            version = None
            if name != "baseline_dummy":
                registered = mlflow.register_model(model_info.model_uri, model_name)
                version = str(registered.version)
                client.set_model_version_tag(model_name, version, "candidate", name)
                client.set_model_version_tag(
                    model_name, version, "dataset.sha256", splits.file_sha256
                )
                client.update_model_version(
                    model_name,
                    version,
                    description=f"{name}; validation macro F1={metrics['f1_macro']:.4f}; "
                    f"trained on train split only; seed={seed}",
                )
            fitted[name] = model
            results.append(
                {
                    "name": name,
                    "run_id": run.info.run_id,
                    "version": version,
                    "model_uri": model_info.model_uri,
                    "fit_seconds": fit_seconds,
                    **metrics,
                }
            )

    # Test labels never participate in ranking or model promotion.
    simplicity = {"logreg_words": 0, "svm_bigrams": 1, "logreg_characters": 2}
    best = sorted(
        (r for r in results if r["version"] is not None),
        key=lambda r: (-r["f1_macro"], simplicity[r["name"]]),
    )[0]
    best_model = fitted[best["name"]]
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_input(datasets["test"], context="testing")
        predicted = best_model.predict(splits.test[["text"]])
        test_metrics = quality_metrics(splits.test["category"], predicted)
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        with tempfile.TemporaryDirectory(prefix="resume-test-") as tmp:
            report = create_diagnostics(splits.test, predicted, Path(tmp))
            mlflow.log_dict(report, "test/classification_report.json")
            mlflow.log_artifacts(tmp, artifact_path="test")
        mlflow.log_dict(results, "comparison/runs.json")
    client.set_registered_model_alias(model_name, alias, best["version"])
    comparison = pd.DataFrame(results)
    comparison["is_champion"] = comparison["name"] == best["name"]
    comparison = comparison.sort_values(["f1_macro", "is_champion"], ascending=[False, False])
    comparison.to_csv(output / "comparison.csv", index=False)
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
    (output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_artifact(str(output / "comparison.csv"), artifact_path="comparison")
        mlflow.log_dict(summary, "comparison/training_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register resume classification models")
    parser.add_argument("--data", type=Path, default=Path("data/cv_target.csv"))
    parser.add_argument(
        "--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    parser.add_argument(
        "--experiment", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "resume-classifier-hw3")
    )
    parser.add_argument("--model-name", default=os.getenv("MLFLOW_MODEL_NAME", "resume-classifier"))
    parser.add_argument("--alias", default=os.getenv("MLFLOW_MODEL_ALIAS", "champion"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--source", help="Stable original CSV URI for dataset provenance")
    args = parser.parse_args()
    summary = run_training(
        args.data,
        args.tracking_uri,
        args.experiment,
        args.model_name,
        args.alias,
        args.seed,
        args.output,
        args.source,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
