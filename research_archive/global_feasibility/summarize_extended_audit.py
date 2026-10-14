"""Validate saved audit manifests and precipitation windows without network access."""
import json
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

P = Path(__file__).resolve().parent


def main():
    import numpy as np
    import rasterio
    weather = json.loads((P / 'extended_weather_manifest.json').read_text())
    result = {'overall_gate': 'FAIL_LABEL_COMPATIBILITY_UNRESOLVED', 'archive_dates': [], 'regional_values': []}
    for day in sorted(set(x['run_id'].split('.')[1].split('/')[0] for x in weather['archive_inventories'])):
        rows = [x for x in weather['archive_inventories'] if day in x['run_id']]
        cutoff = datetime.strptime(day, '%Y%m%d').replace(tzinfo=timezone.utc, hour=11)
        target = cutoff.replace(hour=0) + timedelta(days=1)
        times = [parsedate_to_datetime(x['grib_last_modified']) for x in rows if x.get('grib_last_modified')]
        errors = [x for x in rows if x.get('error') or x.get('missing_fields')]
        valid = all(datetime.fromisoformat(x['valid_utc']) == target + timedelta(hours=x['lead_hours']-18) for x in rows)
        intervals = all([6*((x['lead_hours']-1)//6), x['lead_hours']] in x.get('apcp_intervals_hours', []) for x in rows if x['lead_hours'] >= 19)
        result['archive_dates'].append({'initialization_utc': cutoff.replace(hour=6).isoformat(),
            'target_start_utc': target.isoformat(), 'target_end_exclusive_utc': (target+timedelta(days=1)).isoformat(),
            'inventories': len(rows), 'inventory_errors_or_missing_fields': len(errors),
            'valid_timestamps_align': valid, 'required_accumulations_present': intervals,
            'grib_archive_timestamp_count': len(times), 'latest_grib_archive_last_modified': max(times).isoformat() if times else None,
            'all_archive_object_times_before_cutoff': len(times)==25 and all(t<=cutoff for t in times),
            'actual_pre_cutoff_public_retrieval_proven': False})
    keys = {'temperature_K': (0,0,103,2), 'humidity_percent': (1,1,103,2),
            'u_wind_ms': (2,2,103,10), 'v_wind_ms': (2,3,103,10)}
    for region in weather['regions']:
        rows = [x for x in weather['recent_cutouts'] if x['region']==region]
        errors = [x for x in rows if 'error' in x]
        sampled = {'region': region, 'run_id': 'gfs.20261009/06', 'retrieval_errors': errors,
                   'forecast_files': len(rows), 'ranges': {}, 'missing_field_slots': []}
        by_lead = {x['lead_hours']: x.get('records', []) for x in rows}
        for name, key in keys.items():
            vals=[]
            for lead in range(18,42):
                rec=[r for r in by_lead[lead] if (r['category'],r['parameter'],r['surface_type'],r['surface_value'])==key]
                if len(rec)!=1:
                    sampled['missing_field_slots'].append([lead,name])
                else:
                    vals.extend(rec[0]['values'])
            sampled['ranges'][name] = [min(vals),max(vals)] if vals else None
        def rain(lead, start):
            records=[r for r in by_lead[lead] if r['category']==1 and r['parameter']==8
                     and r['template']==8 and r['lead_start']==start
                     and r['statistical_process']==1 and r['interval_unit']==1
                     and r['interval_hours']==lead-start]
            assert records, f'missing precipitation {region} {start}:{lead}'
            assert all(r['values']==records[0]['values'] for r in records), 'inconsistent duplicates'
            return records[0]['values']
        try:
            hourly=[]
            for lead in range(19,43):
                start=6*((lead-1)//6)
                a=rain(lead,start)
                b=rain(lead-1,start) if lead-1>start else [0.0]*len(a)
                hourly.append([v-u for u,v in zip(b,a)])
            daily=[sum(x) for x in zip(*hourly)]
            c0,c1=rain(18,0),rain(42,0)
            reference=[b-a for a,b in zip(c0,c1)]
            sampled['precipitation']={'hourly_intervals':len(hourly),'point_count':len(daily),
                'daily_mm_range':[min(daily),max(daily)], 'minimum_hourly_increment_mm':min(min(x) for x in hourly),
                'max_difference_vs_cumulative_endpoint_mm':max(abs(a-b) for a,b in zip(daily,reference)),
                'method':'within-block hourly differencing; reset at six-hour boundary; no clipping of negative increments'}
        except Exception as exc:
            sampled['precipitation']={'error':str(exc)}
        result['regional_values'].append(sampled)
    static=json.loads((P/'extended_static_manifest.json').read_text())
    result['new_static_cells']=[{'region':x['region'],'stratum':x['intended_stratum'],'grid_id':x['grid_cell']['id'],
        'elevation_coverage':x['elevation'].get('valid_area_fraction'),'landcover_coverage':x['land_cover'].get('valid_area_fraction'),
        'elevation_mean_m':x['elevation'].get('elevation_m',{}).get('area_weighted_mean'),
        'water_fraction':x['land_cover'].get('class_fractions_of_valid_area',{}).get('permanent_water'),
        'tree_fraction':x['land_cover'].get('class_fractions_of_valid_area',{}).get('tree_cover'),
        'errors':[v['error'] for v in [x['elevation'],x['land_cover']] if 'error' in v]}
        for x in static['samples']]
    result['weather_transfer_bytes']=sum(x.get('index_bytes',0) for x in weather['archive_inventories'])+sum(x.get('bytes',0) for x in weather['recent_cutouts'])+sum(x.get('bytes',0) for x in weather.get('archive_grid_headers',[]))
    checked, max_difference, timestamp_errors = 0, 0.0, []
    for item in weather['recent_cutouts']:
        if 'local_file' not in item:
            continue
        with rasterio.open(P/item['local_file']) as ds:
            assert ds.count == len(item['records'])
            for band, record in enumerate(item['records'],1):
                tags=ds.tags(band)
                actual=ds.read(band).ravel().astype(float)
                if tags.get('GRIB_UNIT')=='[C]':
                    actual+=273.15
                error=float(np.max(np.abs(np.sort(actual)-np.sort(record['values']))))
                max_difference=max(max_difference,error)
                expected=datetime(2026,10,9,6,tzinfo=timezone.utc)+timedelta(hours=item['lead_hours'])
                if int(tags['GRIB_VALID_TIME']) != int(expected.timestamp()):
                    timestamp_errors.append([item['local_file'],band])
                checked+=1
    result['independent_gdal_decode_check']={'bands_checked':checked,'max_numeric_difference':max_difference,
        'timestamp_errors':timestamp_errors,'within_0_0001_native_units':max_difference<0.0001,
        'gdal_version':rasterio.__gdal_version__}
    (P/'extended_audit_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
