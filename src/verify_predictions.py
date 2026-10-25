"""Exercise saved models; add --live for real provider requests.

No training or model updates. Results are local diagnostic evidence.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .firecube_inference import DEFAULT_CUBE, FireCubeExp5Inference
from .firecube_live import HistoricalCubeDynamicProvider
from .place_search import NominatimSearch
from .predict import ImagePredictor, TabularPredictor, default_weather, fuse_probabilities
from .production_features import ProductionFeaturePipeline, ProductionPreprocessor
from .production_model import ProductionPredictionService, XGBoostProductionModel

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Contact place, weather and MODIS services')
    parser.add_argument('--historical', action='store_true', help='Require and verify full-cube historical replay')
    parser.add_argument('--output', type=Path, default=ROOT / '.cache' / 'verification.json')
    args = parser.parse_args()
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'operational_validity': False, 'checks': {}}

    def check(name, action):
        try:
            result = action()
            report['checks'][name] = {'passed': True, 'result': result}
            print(f'PASS {name}', flush=True)
        except Exception as exc:
            report['checks'][name] = {'passed': False, 'error': str(exc)}
            print(f'FAIL {name}: {exc}', flush=True)

    def legacy():
        weather = TabularPredictor().predict_proba(default_weather())
        samples = sorted((ROOT / 'data/images/samples').glob('*/*.jpg'))
        if not samples:
            raise FileNotFoundError('No bundled image samples')
        with Image.open(samples[0]) as image:
            photo = ImagePredictor().predict_proba(image)
        return fuse_probabilities(photo['p_fire'], weather['p_fire'])

    def production(live=False):
        pipeline = ProductionFeaturePipeline(
            dynamic_provider=None if live else HistoricalCubeDynamicProvider(),
            preprocessor=ProductionPreprocessor.load(),
        )
        service = ProductionPredictionService(pipeline, XGBoostProductionModel.load(ROOT / 'models/production/xgb_baseline.joblib'))
        result = service.predict_recent(38, 23.7) if live else service.predict(38, 23.7, '2020-07-15')
        if not 0 <= result['raw_probability'] <= 1:
            raise ValueError('Model returned an invalid score')
        return result

    check('legacy_image_weather_fusion', legacy)
    from .static_geography import CompactStaticFeatureProvider
    check('compact_geography', lambda: CompactStaticFeatureProvider().fetch_static(38, 23.7, 2026)[2])
    if args.historical or DEFAULT_CUBE.exists():
        check('historical_exp5', lambda: FireCubeExp5Inference().predict_historical(38, 23.7, '2020-07-15'))
        check('historical_track_b', production)
    else:
        report['skipped'] = {'historical_exp5': 'optional full cube absent', 'historical_track_b': 'optional full cube absent'}
        print('SKIP historical replay: optional full cube absent')
    if args.live:
        def place():
            matches = NominatimSearch().search('Athens, Greece')
            if not matches:
                raise RuntimeError('No place matches returned')
            return matches[0]
        check('live_place_search', place)
        check('live_track_b', lambda: production(True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Diagnostic evidence: {args.output}')
    return int(any(not item['passed'] for item in report['checks'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
