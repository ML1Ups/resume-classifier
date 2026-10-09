import argparse
import json
import os

from mlflow import MlflowClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect versions or move a model alias")
    parser.add_argument(
        "--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
    )
    parser.add_argument("--model-name", default="resume-classifier")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--version", help="If omitted, only inspect the Registry")
    args = parser.parse_args()
    client = MlflowClient(tracking_uri=args.tracking_uri)
    if args.version:
        client.set_registered_model_alias(args.model_name, args.alias, args.version)
    versions = client.search_model_versions(f"name='{args.model_name}'")
    print(
        json.dumps(
            {
                "name": args.model_name,
                "aliases": client.get_registered_model(args.model_name).aliases,
                "versions": [
                    {
                        "version": version.version,
                        "run_id": version.run_id,
                        "candidate": version.tags.get("candidate"),
                    }
                    for version in versions
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
