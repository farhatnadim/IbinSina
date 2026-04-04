#!/usr/bin/env python3
"""
Sync offline MLflow runs to the server.

When training runs in offline mode (MLflow server unavailable), data is saved
to JSON files in the mlflow_offline/ directory. This script uploads those
runs to the MLflow server once it's available.

Usage:
    python scripts/sync_mlflow_offline.py
    python scripts/sync_mlflow_offline.py --offline-dir /path/to/offline
    python scripts/sync_mlflow_offline.py --dry-run  # Preview without uploading
    python scripts/sync_mlflow_offline.py --keep     # Don't delete after sync
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime


def sync_run(mlflow, run_data: dict, experiment_name: str, dry_run: bool = False) -> bool:
    """
    Sync a single offline run to MLflow.

    Args:
        mlflow: mlflow module
        run_data: Dictionary containing the offline run data
        experiment_name: Name of the MLflow experiment
        dry_run: If True, just print what would happen

    Returns:
        True if sync successful, False otherwise
    """
    run_name = run_data.get('run_name', 'unnamed')
    tags = run_data.get('tags', {})
    params = run_data.get('params', {})
    metrics = run_data.get('metrics', [])
    artifacts = run_data.get('artifacts', [])
    nested_runs = run_data.get('nested_runs', [])

    if dry_run:
        print(f"  Would create run: {run_name}")
        print(f"    Tags: {len(tags)}")
        print(f"    Params: {len(params)}")
        print(f"    Metric entries: {len(metrics)}")
        print(f"    Artifacts: {len(artifacts)}")
        print(f"    Nested runs: {len(nested_runs)}")
        return True

    try:
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name, tags=tags):
            # Log parameters
            if params:
                # MLflow doesn't like None values
                clean_params = {
                    k: (str(v) if v is not None else "None")
                    for k, v in params.items()
                }
                mlflow.log_params(clean_params)

            # Log metrics with steps
            for entry in metrics:
                step = entry.get('step')
                values = entry.get('values', {})
                if values:
                    mlflow.log_metrics(values, step=step)

            # Log artifacts (only if they still exist)
            for artifact_path in artifacts:
                if Path(artifact_path).exists():
                    mlflow.log_artifact(artifact_path)
                else:
                    print(f"    Warning: Artifact not found: {artifact_path}")

            # Handle nested runs (e.g., CV folds)
            for nested in nested_runs:
                nested_name = nested.get('name', 'nested')
                nested_tags = nested.get('tags', {})
                nested_params = nested.get('params', {})
                nested_metrics = nested.get('metrics', [])

                with mlflow.start_run(run_name=nested_name, nested=True, tags=nested_tags):
                    if nested_params:
                        clean_nested_params = {
                            k: (str(v) if v is not None else "None")
                            for k, v in nested_params.items()
                        }
                        mlflow.log_params(clean_nested_params)

                    for entry in nested_metrics:
                        step = entry.get('step')
                        values = entry.get('values', {})
                        if values:
                            mlflow.log_metrics(values, step=step)

        return True

    except Exception as e:
        print(f"    Error syncing run: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Sync offline MLflow runs to server'
    )
    parser.add_argument(
        '--offline-dir',
        type=str,
        default='mlflow_offline',
        help='Directory containing offline run JSON files (default: mlflow_offline)',
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='mil-training',
        help='MLflow experiment name (default: mil-training)',
    )
    parser.add_argument(
        '--tracking-uri',
        type=str,
        default=None,
        help='MLflow tracking URI (default: from MLFLOW_TRACKING_URI env)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be synced without actually uploading',
    )
    parser.add_argument(
        '--keep',
        action='store_true',
        help='Keep offline files after successful sync (default: delete them)',
    )
    args = parser.parse_args()

    offline_dir = Path(args.offline_dir)

    if not offline_dir.exists():
        print(f"Offline directory not found: {offline_dir}")
        print("Nothing to sync.")
        return 0

    # Find all offline run files
    offline_files = sorted(offline_dir.glob('run_*.json'))

    if not offline_files:
        print(f"No offline runs found in {offline_dir}")
        return 0

    print(f"Found {len(offline_files)} offline run(s) to sync")
    print()

    if args.dry_run:
        print("=== DRY RUN MODE ===")
        print()

    # Import and configure MLflow
    if not args.dry_run:
        try:
            import mlflow
        except ImportError:
            print("Error: mlflow package not installed")
            print("Install with: pip install mlflow")
            return 1

        tracking_uri = args.tracking_uri or os.environ.get('MLFLOW_TRACKING_URI')
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            print(f"MLflow tracking URI: {tracking_uri}")
        else:
            print("Warning: No tracking URI set, using default (local)")

        # Test connection
        try:
            mlflow.set_experiment(args.experiment)
            print(f"MLflow experiment: {args.experiment}")
            print()
        except Exception as e:
            print(f"Error connecting to MLflow: {e}")
            print("Make sure the MLflow server is running.")
            return 1
    else:
        mlflow = None

    # Process each offline file
    synced = 0
    failed = 0

    for filepath in offline_files:
        print(f"Processing: {filepath.name}")

        try:
            with open(filepath, 'r') as f:
                run_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  Error reading JSON: {e}")
            failed += 1
            continue

        success = sync_run(
            mlflow=mlflow,
            run_data=run_data,
            experiment_name=args.experiment,
            dry_run=args.dry_run,
        )

        if success:
            synced += 1
            if not args.dry_run and not args.keep:
                filepath.unlink()
                print(f"  Synced and deleted: {filepath.name}")
            elif not args.dry_run:
                print(f"  Synced (kept): {filepath.name}")
        else:
            failed += 1

        print()

    # Summary
    print("=" * 50)
    print(f"Synced: {synced}")
    print(f"Failed: {failed}")

    if args.dry_run:
        print()
        print("This was a dry run. No data was uploaded.")
        print("Run without --dry-run to actually sync.")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
