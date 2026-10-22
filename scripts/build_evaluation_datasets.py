import argparse
import json
from hashlib import sha256
from pathlib import Path

from sklearn.model_selection import train_test_split

from resume_classifier.ml.dataset import load_and_split


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_split(frame, path: Path) -> None:
    frame[["record_id", "text", "category"]].to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create reproducible v1/v2 train/validation/test datasets from cv_target.csv"
    )
    parser.add_argument("--source", type=Path, default=Path("data/cv_target.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--v2-train-fraction", type=float, default=0.1)
    args = parser.parse_args()

    splits = load_and_split(args.source, seed=args.seed)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    save_split(splits.full, output / "full.csv")
    save_split(splits.train, output / "v1_train.csv")
    save_split(splits.validation, output / "v1_validation.csv")
    save_split(splits.test, output / "test_v1.csv")

    v2_train, _ = train_test_split(
        splits.train,
        train_size=args.v2_train_fraction,
        random_state=args.seed,
        stratify=splits.train["category"],
    )
    save_split(v2_train, output / "v2_train.csv")
    save_split(splits.validation, output / "v2_validation.csv")
    save_split(splits.test, output / "test_v2.csv")

    manifest = {
        "format": "resume-classifier-versioned-evaluation",
        "source": splits.source,
        "source_sha256": splits.file_sha256,
        "seed": args.seed,
        "base_split": {
            "strategy": "stratified_group",
            "train_fraction": len(splits.train) / len(splits.full),
            "validation_fraction": len(splits.validation) / len(splits.full),
            "test_fraction": len(splits.test) / len(splits.full),
        },
        "preparation": {
            **splits.preparation,
            "split_strategy": "stratified_group",
            "version_policy": (
                "v1 uses the complete train split; v2 uses a stratified 10% subset of "
                "the same train split. Validation and test are identical between versions."
            ),
        },
        "versions": {
            "v1": {
                "train_rows": len(splits.train),
                "validation_rows": len(splits.validation),
                "test_rows": len(splits.test),
            },
            "v2": {
                "train_rows": len(v2_train),
                "validation_rows": len(splits.validation),
                "test_rows": len(splits.test),
            },
        },
        "files": {
            "full": file_digest(output / "full.csv"),
            "v1_train": file_digest(output / "v1_train.csv"),
            "v1_validation": file_digest(output / "v1_validation.csv"),
            "test_v1": file_digest(output / "test_v1.csv"),
            "v2_train": file_digest(output / "v2_train.csv"),
            "v2_validation": file_digest(output / "v2_validation.csv"),
            "test_v2": file_digest(output / "test_v2.csv"),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest["versions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
