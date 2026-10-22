import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from resume_classifier.ml.prepare import normalize_text, prepare_cv_export


@dataclass(frozen=True)
class DatasetSplits:
    full: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    file_sha256: str
    source: str
    seed: int
    preparation: dict = field(default_factory=dict)
    dataset_version: str = "source"


def load_and_split(path: Path, seed: int = 42, source: str | None = None) -> DatasetSplits:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "professionList" in frame.columns:
        frame, preparation = prepare_cv_export(frame)
        return split_groups(
            frame, digest.hexdigest(), source or path.resolve().as_uri(), seed, preparation
        )
    if not {"text", "category"}.issubset(frame.columns):
        raise ValueError("CSV must contain text and category columns")
    frame = frame[["text", "category"]].copy()
    frame["text"] = frame["text"].str.strip()
    frame["category"] = frame["category"].str.strip()
    if frame.empty or (frame["text"] == "").any() or (frame["category"] == "").any():
        raise ValueError("Dataset must have non-empty texts and categories")
    normalized = frame["text"].map(normalize_text)
    if normalized.duplicated().any():
        raise ValueError("Duplicate normalized texts: remove duplicates before splitting")
    counts = frame["category"].value_counts()
    if len(counts) < 2 or counts.min() < 10:
        raise ValueError("Need at least two categories and at least 10 rows in each category")
    frame.insert(0, "record_id", normalized.map(lambda t: sha256(t.encode()).hexdigest()[:16]))
    train_val, test = train_test_split(
        frame,
        test_size=0.2,
        random_state=seed,
        stratify=frame["category"],
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        random_state=seed,
        stratify=train_val["category"],
    )
    return DatasetSplits(
        full=frame,
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        file_sha256=digest.hexdigest(),
        source=source or path.resolve().as_uri(),
        seed=seed,
        preparation={
            "format": "flat text/category CSV",
            "raw_rows": len(frame),
            "prepared_rows": len(frame),
            "synthetic_demo": path.name == "resumes_demo.csv",
        },
    )


def load_versioned_split(directory: Path, version: str, seed: int = 42) -> DatasetSplits:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    full = pd.read_csv(directory / "full.csv", dtype=str, keep_default_na=False)
    train = pd.read_csv(directory / f"{version}_train.csv", dtype=str, keep_default_na=False)
    validation = pd.read_csv(
        directory / f"{version}_validation.csv",
        dtype=str,
        keep_default_na=False,
    )
    test = pd.read_csv(directory / f"test_{version}.csv", dtype=str, keep_default_na=False)
    required = {"record_id", "text", "category"}
    for name, frame in {
        "full": full,
        "train": train,
        "validation": validation,
        "test": test,
    }.items():
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} split must contain record_id, text and category columns")

    return DatasetSplits(
        full=full,
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        file_sha256=manifest["source_sha256"],
        source=manifest["source"],
        seed=seed,
        preparation={
            **manifest["preparation"],
            "dataset_version": version,
            "base_split": manifest["base_split"],
        },
        dataset_version=version,
    )


def load_evaluation_test(directory: str, version: str | None) -> pd.DataFrame:
    if version is None:
        raise ValueError("Dataset version is required for evaluation")
    path = Path(directory) / f"test_{version}.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frame[["record_id", "text", "category"]]


def split_groups(
    frame: pd.DataFrame, file_digest: str, source: str, seed: int, preparation: dict
) -> DatasetSplits:
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    development_ids, test_ids = next(outer.split(frame, frame["category"], frame["group_id"]))
    development = frame.iloc[development_ids]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    train_ids, validation_ids = next(
        inner.split(
            development,
            development["category"],
            development["group_id"],
        )
    )
    parts = [development.iloc[train_ids], development.iloc[validation_ids], frame.iloc[test_ids]]
    expected = set(frame["category"])
    if any(set(part["category"]) != expected for part in parts):
        raise ValueError(
            "A class is absent from a grouped split; review class support/group policy"
        )
    return DatasetSplits(
        frame,
        *(part.reset_index(drop=True) for part in parts),
        file_digest,
        source,
        seed,
        preparation,
    )


def lineage(splits: DatasetSplits, digests: dict[str, str]) -> dict:
    parts = {
        "full": splits.full,
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    return {
        "source": splits.source,
        "file_sha256": splits.file_sha256,
        "parent_dataset_digest": digests["full"],
        "dataset_version": splits.dataset_version,
        "split": {
            "strategy": splits.preparation.get(
                "split_strategy",
                "stratified_group" if "group_id" in splits.full else "stratified",
            ),
            "seed": splits.seed,
            "train_fraction": len(splits.train) / len(splits.full),
            "validation_fraction": len(splits.validation) / len(splits.full),
            "test_fraction": len(splits.test) / len(splits.full),
            "base_train_fraction": 0.6,
            "base_validation_fraction": 0.2,
            "base_test_fraction": 0.2,
        },
        "preparation": splits.preparation,
        "parts": {
            name: {
                "rows": len(frame),
                "digest": digests[name],
                "record_ids": frame["record_id"].tolist(),
                "actual_fraction": len(frame) / len(splits.full),
                **(
                    {"group_ids": sorted(frame["group_id"].unique())} if "group_id" in frame else {}
                ),
            }
            for name, frame in parts.items()
        },
    }
