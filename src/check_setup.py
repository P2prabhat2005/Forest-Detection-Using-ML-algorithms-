"""Read-only setup check: python -m src.check_setup."""
from pathlib import Path
import argparse
import importlib
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical', action='store_true', help='Also require the full local cube for historical replay')
    args = parser.parse_args()
    failed = False
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    for name in ('streamlit', 'numpy', 'pandas', 'sklearn', 'xgboost', 'netCDF4', 'torch', 'torchvision'):
        try:
            importlib.import_module(name)
            print(f"OK dependency: {name}")
        except Exception as exc:
            failed = True
            print(f"FAIL dependency: {name}: {exc}")
    for relative in (
        'models/production/xgb_baseline.joblib',
        'models/production/feature_preprocessor.joblib',
        'models/production/feature_schema.json',
        'results/firecube_direct_10km_artifact_rerun/xgb_direct_10km.joblib',
        'results/firecube_direct_10km_artifact_rerun/feature_aggregation.json',
        'models/image_mobilenet.pt', 'models/tabular_rf.joblib',
    ):
        exists = (ROOT / relative).is_file() and (ROOT / relative).stat().st_size > 0
        failed |= not exists
        print(f"{'OK' if exists else 'FAIL'} asset: {relative}")
    try:
        from .static_geography import CompactStaticFeatureProvider
        provider = CompactStaticFeatureProvider()
        print(f'OK compact geography: {provider.path} (SHA-256 {provider.checksum})')
    except Exception as exc:
        failed = True
        print(f'FAIL compact geography: {exc}')
    cube = ROOT / 'data/raw/firecube/dataset_greece.nc'
    if args.historical:
        exists = cube.is_file() and cube.stat().st_size > 0
        failed |= not exists
        print(f"{'OK' if exists else 'FAIL'} historical cube: {cube}")
    else:
        print('INFO full FireCube cube is optional for live Track-B; required for historical replay.')
    if failed:
        print('See README.md for environment setup and authorized FireCube provisioning.')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
