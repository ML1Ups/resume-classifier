import argparse
import json
from pathlib import Path

import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

from resume_classifier.logs import configure_logging
from resume_classifier.ml.dataset import file_sha256, load_and_split

SEED = 42
V2_TRAIN_FRACTION = 0.1

logger = structlog.get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset versions v1 and v2 from a CSV")
    parser.add_argument("--source", type=Path, default=Path("data/cv_target.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation"))
    args = parser.parse_args()
    configure_logging("INFO", json_logs=True)

    splits = load_and_split(args.source, SEED)
    v2_train, _ = train_test_split(
        splits.train,
        train_size=V2_TRAIN_FRACTION,
        random_state=SEED,
        stratify=splits.train["category"],
    )
    parts: dict[str, pd.DataFrame] = {
        "full": splits.full,
        "v1_train": splits.train,
        "v1_validation": splits.validation,
        "v1_test": splits.test,
        "v2_train": v2_train,
        "v2_validation": splits.validation,
        "v2_test": splits.test,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in parts.items():
        frame[["record_id", "text", "category"]].to_csv(args.output / f"{name}.csv", index=False)

    versions = {
        version: {part: len(parts[f"{version}_{part}"]) for part in ("train", "validation", "test")}
        for version in ("v1", "v2")
    }
    manifest = {
        "source": splits.source,
        "source_sha256": splits.file_sha256,
        "seed": SEED,
        "base_split": {
            "strategy": "stratified_group",
            "train_fraction": len(splits.train) / len(splits.full),
            "validation_fraction": len(splits.validation) / len(splits.full),
            "test_fraction": len(splits.test) / len(splits.full),
        },
        "preparation": {
            **splits.preparation,
            "split_strategy": "stratified_group",
            "version_policy": "v1 uses the whole train split, v2 a stratified "
            f"{V2_TRAIN_FRACTION:.0%} of it; validation and test are the same",
        },
        "versions": versions,
        "files": {name: file_sha256(args.output / f"{name}.csv") for name in parts},
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("dataset_versions_built", output=str(args.output), versions=versions)


if __name__ == "__main__":
    main()
