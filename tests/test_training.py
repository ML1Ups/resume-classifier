import json
from pathlib import Path
from unittest.mock import Mock

import mlflow
import pytest
from mlflow import MlflowClient

from resume_classifier.ml.dataset import load_and_split
from resume_classifier.ml.train import run_training

DEMO = Path(__file__).resolve().parents[1] / "data/resumes_demo.csv"


def test_training_tracks_lineage_logs_models_and_selects_on_validation(tmp_path):
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    previous_tracking = mlflow.get_tracking_uri()
    previous_registry = mlflow.get_registry_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.create_experiment("training-test", artifact_location=(tmp_path / "artifacts").as_uri())
    try:
        summary = run_training(DEMO, uri, experiment="training-test", output=tmp_path / "results")
        client = MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name("training-test")
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) == 5
        versions = client.search_model_versions("name='resume-classifier'")
        assert len(versions) == 3
        best = summary["champion"]
        assert best["f1_macro"] == max(r["f1_macro"] for r in summary["runs"] if r["version"])
        selected = client.get_model_version_by_alias("resume-classifier", "champion")
        assert str(selected.version) == best["version"]
        assert selected.run_id == best["run_id"]
        champion_run = client.get_run(best["run_id"])
        contexts = {tag.value for item in champion_run.inputs.dataset_inputs for tag in item.tags}
        assert {"training", "validation", "testing"}.issubset(contexts)
        assert "test_f1_macro" in champion_run.data.metrics
        other = next(r for r in summary["runs"] if r["version"] and r["name"] != best["name"])
        assert "test_f1_macro" not in client.get_run(other["run_id"]).data.metrics
        loaded = mlflow.sklearn.load_model(f"models:/resume-classifier/{selected.version}")
        split = load_and_split(DEMO)
        assert len(loaded.predict(split.test[["text"]])) == 48
        path = client.download_artifacts(best["run_id"], "data/lineage.json")
        provenance = json.loads(Path(path).read_text())
        assert provenance["file_sha256"] == split.file_sha256
        assert provenance["parts"]["train"]["rows"] == 144
        assert (tmp_path / "results/comparison.csv").is_file()
    finally:
        mlflow.set_tracking_uri(previous_tracking)
        mlflow.set_registry_uri(previous_registry)


def test_training_cli_passes_explicit_arguments(monkeypatch, tmp_path):
    import resume_classifier.ml.train as module

    runner = Mock(return_value={"ok": True})
    monkeypatch.setattr(module, "run_training", runner)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--data",
            str(DEMO),
            "--output",
            str(tmp_path),
            "--tracking-uri",
            "http://localhost:5000",
        ],
    )
    module.main()
    assert runner.call_args.args[0] == DEMO
    assert runner.call_args.args[1] == "http://localhost:5000"


@pytest.mark.parametrize("version", [None, "2"])
def test_registry_cli_inspects_and_moves_alias(monkeypatch, version):
    import resume_classifier.ml.registry as module

    client = Mock()
    client.get_registered_model.return_value.aliases = {"champion": "2"}
    client.search_model_versions.return_value = []
    monkeypatch.setattr(module, "MlflowClient", Mock(return_value=client))
    args = ["registry"] if version is None else ["registry", "--version", version]
    monkeypatch.setattr("sys.argv", args)
    module.main()
    if version:
        client.set_registered_model_alias.assert_called_once_with(
            "resume-classifier", "champion", "2"
        )
    else:
        client.set_registered_model_alias.assert_not_called()
