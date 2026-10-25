"""Evaluate the unchanged Track-B artifacts on archived temporal holdouts.

These samples use artificial prevalence. This is not live/provider validation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .production_features import ProductionPreprocessor, raw_training_features
from .production_model import XGBoostProductionModel

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=ROOT / 'research_archive/research_data/processed/firecube_samples.npz')
    parser.add_argument('--output', type=Path, default=ROOT / 'models/production/sampled_temporal_evaluation.json')
    args = parser.parse_args()
    raw, labels, years = raw_training_features(args.samples)
    preprocessor = ProductionPreprocessor.load()
    model = XGBoostProductionModel.load(ROOT / 'models/production/xgb_baseline.joblib')
    report = {
        'model': 'models/production/xgb_baseline.joblib',
        'samples': str(args.samples.relative_to(ROOT)) if args.samples.is_relative_to(ROOT) else str(args.samples),
        'training_years': '<=2018',
        'limitations': 'Artificial sampled prevalence; historical FireCube inputs; not live/provider or geographic validation. No retraining, calibration or threshold selection.',
        'operational_validity': False,
        'years': {},
    }
    for year in (2019, 2020, 2021):
        mask = years == year
        y = labels[mask]
        if len(y) == 0 or len(np.unique(y)) != 2:
            raise ValueError(f'{year} requires both positive and negative samples')
        features, missing = preprocessor.transform(raw[mask])
        scores = model.estimator.predict_proba(features)[:, 1]
        report['years'][str(year)] = {
            'role': 'validation' if year == 2019 else 'temporal test',
            'n_samples': int(len(y)), 'positive_rate': float(y.mean()),
            'roc_auc': float(roc_auc_score(y, scores)),
            'average_precision': float(average_precision_score(y, scores)),
            'brier_score': float(brier_score_loss(y, scores)),
            'imputed_feature_count': int(missing.sum()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
