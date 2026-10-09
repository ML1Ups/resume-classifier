from pathlib import Path

import pandas as pd
import pytest

from resume_classifier.ml.dataset import lineage, load_and_split

DEMO = Path(__file__).resolve().parents[1] / "data/resumes_demo.csv"


def test_splits_are_disjoint_stratified_and_reproducible():
    splits = load_and_split(DEMO)
    repeat = load_and_split(DEMO)
    assert (len(splits.train), len(splits.validation), len(splits.test)) == (144, 48, 48)
    assert splits.train.equals(repeat.train)
    ids = [set(frame.record_id) for frame in [splits.train, splits.validation, splits.test]]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert set.union(*ids) == set(splits.full.record_id)
    assert splits.train.category.value_counts().tolist() == [24] * 6
    assert len(splits.file_sha256) == 64
    provenance = lineage(splits, dict.fromkeys(["full", "train", "validation", "test"], "digest"))
    assert provenance["parts"]["train"]["record_ids"] == splits.train.record_id.tolist()
    assert provenance["parent_dataset_digest"] == "digest"


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"x": [1]}), "columns"),
        (pd.DataFrame({"text": [" "], "category": ["a"]}), "non-empty"),
        (
            pd.DataFrame({"text": ["Hello world", " hello   WORLD "], "category": ["a", "b"]}),
            "Duplicate",
        ),
        (pd.DataFrame({"text": ["one"], "category": ["a"]}), "at least"),
    ],
)
def test_invalid_datasets_fail_before_training(tmp_path, frame, message):
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_and_split(path)
