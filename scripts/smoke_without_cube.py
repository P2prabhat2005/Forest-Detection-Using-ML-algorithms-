"""Stage the current source/artifacts without ignored data and run offline checks.

Uses the calling Python's installed packages; does not claim a fresh dependency
installation. Leaves its small staging directory and logs under .cache.
Run: python -m scripts.smoke_without_cube [--live]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Also contact the actual live providers')
    args = parser.parse_args()
    stage = ROOT / '.cache' / ('no-cube-checkout-' + uuid.uuid4().hex[:8])
    stage.mkdir(parents=True)
    paths = subprocess.check_output(['git', 'ls-files', '-co', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    for name in sorted(set(paths) - {''}):
        source, target = ROOT / name, stage / name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    assert not (stage / 'data/raw/firecube/dataset_greece.nc').exists()
    assert not (stage / 'research_archive/research_data/processed/firecube_samples.npz').exists()
    temp = stage / '.cache/temp'
    temp.mkdir(parents=True)
    env = dict(os.environ, TEMP=str(temp), TMP=str(temp), PYTHONPATH=str(stage), PYTHONUTF8='1')
    report = {'stage': str(stage), 'python': sys.executable, 'cube_present': False,
              'dependency_installation': 'reuses calling environment; no package installation performed', 'checks': {}}
    commands = {
        'setup': ['-m', 'src.check_setup'],
        'tests': ['-m', 'pytest', '-q', '--tb=short'],
        'verification': ['-m', 'src.verify_predictions'] + (['--live'] if args.live else []),
    }
    for name, command in commands.items():
        result = subprocess.run([sys.executable, *command], cwd=stage, env=env, text=True,
                                encoding='utf-8', errors='replace', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = stage / '.cache' / f'{name}.log'
        log.write_text(result.stdout, encoding='utf-8')
        report['checks'][name] = {'passed': result.returncode == 0, 'exit_code': result.returncode, 'log': str(log)}
        print(f"{'PASS' if result.returncode == 0 else 'FAIL'} {name}: {log}", flush=True)
        print(result.stdout[-2000:], flush=True)
    report_path = ROOT / '.cache/no_cube_smoke.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Report: {report_path}', flush=True)
    return int(any(not check['passed'] for check in report['checks'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
