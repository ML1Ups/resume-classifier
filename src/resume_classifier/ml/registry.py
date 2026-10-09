import argparse
import json
import os

from mlflow import MlflowClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect versions or move a model alias")
    parser.add_argument(
        "--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    parser.add_argument("--model-name", default=os.getenv("MLFLOW_MODEL_NAME", "resume-classifier"))
    parser.add_argument("--alias", default=os.getenv("MLFLOW_MODEL_ALIAS", "champion"))
    parser.add_argument("--version", help="If omitted, only inspect the Registry")
    args = parser.parse_args()
    client = MlflowClient(tracking_uri=args.tracking_uri, registry_uri=args.tracking_uri)
    if args.version:
        client.get_model_version(args.model_name, args.version)
        client.set_registered_model_alias(args.model_name, args.alias, args.version)
    model = client.get_registered_model(args.model_name)
    versions = client.search_model_versions()
    print(
        json.dumps(
            {
                "name": args.model_name,
                "aliases": model.aliases,
                "versions": [
                    {
                        "version": v.version,
                        "run_id": v.run_id,
                        "status": v.status,
                        "candidate": v.tags.get("candidate"),
                    }
                    for v in versions
                    if v.name == args.model_name
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
