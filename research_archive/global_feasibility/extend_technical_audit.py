"""Small audit only: indices/HEADs for four dates, tiny recent cutouts, four static cells.
Reuses completed static processing without rewriting prior reports or manifests.
Python 3.12, numpy, rasterio, pyproj; execute from any working directory.
"""
import concurrent.futures as cf
import hashlib
import json
import re
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
DATES = ['20250115', '20250715', '20260415', '20261009']
FIELDS = [('TMP', '2 m above ground'), ('RH', '2 m above ground'),
          ('UGRD', '10 m above ground'), ('VGRD', '10 m above ground'), ('APCP', 'surface')]
REGIONS = {'Greece': [22.25, 39.25, 22.75, 39.75], 'British Columbia': [-120.75, 50.25, -120.25, 50.75]}


def fetch(url, head=False, cap=150000):
    with urlopen(Request(url, method='HEAD' if head else 'GET'), timeout=45) as r:
        body = b'' if head else r.read(cap + 1)
        if len(body) > cap:
            raise ValueError('response exceeds small-sample limit')
        return body, dict(r.headers), r.status


def inventory(item):
    day, lead = item
    root = f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{day}/06/atmos/gfs.t06z.pgrb2.0p25.f{lead:03}'
    out = {'run_id': f'gfs.{day}/06', 'lead_hours': lead, 'url': root,
           'valid_utc': (datetime.strptime(day, '%Y%m%d').replace(tzinfo=timezone.utc, hour=6) + timedelta(hours=lead)).isoformat()}
    try:
        body, h, status = fetch(root + '.idx')
        _, gh, gs = fetch(root, head=True)
        selected = [line for line in body.decode().splitlines()
                    if any(f':{v}:{level}:' in line for v, level in FIELDS)]
        out.update(http=status, grib_http=gs, index_bytes=len(body), index_sha256=hashlib.sha256(body).hexdigest(),
                   index_last_modified=h.get('Last-Modified'), grib_last_modified=gh.get('Last-Modified'),
                   grib_etag=gh.get('ETag'), index_records=selected,
                   missing_fields=[v for v, level in FIELDS if not any(f':{v}:{level}:' in x for x in selected)],
                   initialization_matches=all(f'd={day}06:' in x for x in selected),
                   instantaneous_leads_match=all(f':{lead} hour fcst:' in x for x in selected if ':APCP:' not in x))
        out['apcp_intervals_hours'] = [list(map(int, m.groups())) for x in selected if ':APCP:' in x
                                      for m in [re.search(r':(\d+)-(\d+) hour acc fcst:', x)] if m]
    except Exception as exc:
        out['error'] = str(exc)
    return out


def decode(data):
    """Strict GRIB2 simple-packing decoder for NOMADS cutouts; fail on other templates."""
    out, start = [], 0
    while start < len(data):
        assert data[start:start+4] == b'GRIB'
        length = int.from_bytes(data[start+8:start+16], 'big')
        pos, sections = start + 16, {}
        while pos < start + length - 4:
            size = int.from_bytes(data[pos:pos+4], 'big')
            sections[data[pos+4]] = data[pos:pos+size]
            pos += size
        p, d, bitmap, payload = sections[4], sections[5], sections[6], sections[7][5:]
        assert int.from_bytes(d[9:11], 'big') == 0, 'unsupported packing'
        assert bitmap[5] == 255, 'bitmap needs explicit decode'
        count, bits = int.from_bytes(d[5:9], 'big'), d[19]
        signed = lambda b: (-1 if b[0] & 128 else 1) * (((b[0] & 127) << 8) + b[1])
        ref, e, dec = struct.unpack('>f', d[11:15])[0], signed(d[15:17]), signed(d[17:19])
        packed = int.from_bytes(payload, 'big')
        vals = [(ref + (((packed >> (len(payload)*8-(i+1)*bits)) & ((1 << bits)-1)) if bits else 0) * 2**e) * 10**(-dec) for i in range(count)]
        row = {'category': p[9], 'parameter': p[10], 'surface_type': p[22],
               'surface_value': int.from_bytes(p[24:28], 'big'), 'point_count': count,
               'lead_start': int.from_bytes(p[18:22], 'big'), 'template': int.from_bytes(p[7:9], 'big'),
               'bitmap_missing': False, 'min': min(vals), 'max': max(vals), 'values': vals,
               'grid_template': int.from_bytes(sections[3][12:14], 'big'),
               'centre': int.from_bytes(sections[1][5:7], 'big'), 'master_table': sections[1][9], 'local_table': sections[1][10]}
        if row['template'] == 8:
            row.update(statistical_process=p[46], interval_unit=p[48], interval_hours=int.from_bytes(p[49:53], 'big'),
                       end_utc=datetime(int.from_bytes(p[34:36], 'big'), *p[36:41], tzinfo=timezone.utc).isoformat())
        out.append(row)
        start += length
    return out


def grid_header(inventory_row):
    """512-byte partial GET of an archived TMP message, never its global field."""
    line = next(x for x in inventory_row['index_records'] if ':TMP:2 m above ground:' in x)
    start = int(line.split(':')[1])
    request = Request(inventory_row['url'], headers={'Range': f'bytes={start}-{start+511}'})
    with urlopen(request, timeout=30) as response:
        assert response.status == 206, 'server did not honor range'
        data = response.read(513)
        assert len(data) == 512
    assert data[:4] == b'GRIB'
    pos = 16
    while data[pos+4] != 3:
        pos += int.from_bytes(data[pos:pos+4], 'big')
    g = data[pos:pos+int.from_bytes(data[pos:pos+4], 'big')]
    def coordinate(offset):
        raw=int.from_bytes(g[offset:offset+4],'big')
        return (-1 if raw & 0x80000000 else 1)*(raw & 0x7fffffff)/1e6
    return {'run_id':inventory_row['run_id'],'url':inventory_row['url'],'range_start':start,'bytes':512,
            'http':206,'sha256':hashlib.sha256(data).hexdigest(),
            'grid_template':int.from_bytes(g[12:14],'big'),'nx':int.from_bytes(g[30:34],'big'),
            'ny':int.from_bytes(g[34:38],'big'),'first_lat':coordinate(46),'first_lon':coordinate(50),
            'last_lat':coordinate(55),'last_lon':coordinate(59),
            'numeric_regional_values_checked':False}


def cutout(item):
    region, lead = item
    west, south, east, north = REGIONS[region]
    params = dict(file=f'gfs.t06z.pgrb2.0p25.f{lead:03}', dir='/gfs.20261009/06/atmos',
                  subregion='', leftlon=west, rightlon=east, bottomlat=south, toplat=north)
    params.update({f'var_{v}': 'on' for v, _ in FIELDS})
    params.update({'lev_2_m_above_ground': 'on', 'lev_10_m_above_ground': 'on', 'lev_surface': 'on'})
    url = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?' + urlencode(params)
    result = {'region': region, 'lead_hours': lead, 'url': url}
    try:
        body, headers, status = fetch(url, cap=64000)
        records = decode(body)
        dest = HERE / 'audit_weather_cutouts' / f'{region.replace(" ", "_")}_20261009_06_f{lead:03}.grib2'
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(body)
        result.update(http=status, bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
                      local_file=str(dest.relative_to(HERE)), records=records)
    except Exception as exc:
        result['error'] = str(exc)
    return result


def main():
    manifest = {'retrieval_utc': datetime.now(timezone.utc).isoformat(), 'overall_gate': 'FAIL_LABEL_COMPATIBILITY_UNRESOLVED',
        'contract': {'issue': '12:00 UTC', 'cutoff': '11:00 UTC', 'target': 'following UTC day [00:00,24:00)',
                     'candidate_run': 'same-day 06Z; selection conditional on proven availability',
                     'instantaneous_leads': '18..41 inclusive (hourly)', 'precipitation_end_leads': '19..42 inclusive'},
        'units': {'TMP_2m': 'K', 'RH_2m': '%', 'UGRD_10m': 'm/s', 'VGRD_10m': 'm/s', 'APCP': 'kg/m2 = mm water'},
        'timing_warning': 'S3 Last-Modified is archive-object metadata, not independent proof of first public availability. No cutoff PASS inferred.',
        'regions': REGIONS}
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        manifest['archive_inventories'] = list(pool.map(inventory, [(d, h) for d in DATES for h in range(18, 43)]))
    print('Archive inventories complete', flush=True)
    manifest['archive_grid_headers'] = [grid_header(x) for x in manifest['archive_inventories'] if x['lead_hours']==18 and 'error' not in x]
    # Serial requests keep pressure on the NOMADS subset service low.
    manifest['recent_cutouts'] = [cutout((region, h)) for region in REGIONS for h in range(18, 43)]
    (HERE / 'extended_weather_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Weather manifest saved', flush=True)

    import compute_static_sample as static
    old = json.loads((HERE / 'static_sample_manifest.json').read_text(encoding='utf-8'))
    new = {'retrieval_utc': datetime.now(timezone.utc).isoformat(), 'overall_gate': manifest['overall_gate'],
           'grid': old['grid'], 'sources': old['sources'], 'processing': old['processing'],
           'reused_samples': 'static_sample_manifest.json (two completed cells, not recomputed)', 'samples': []}
    sites = [('Greece', 'mountain', 21.5, 39.7), ('Greece', 'coastal', 23.6, 38.4),
             ('British Columbia', 'mountain', -122.5, 50.7), ('British Columbia', 'coastal', -123.7, 49.5)]
    for region, stratum, lon, lat in sites:
        cell = static.grid_cell(lon, lat)
        ns = lambda n: f'{"N" if n >= 0 else "S"}{abs(n):02}'
        ew = lambda n: f'{"E" if n >= 0 else "W"}{abs(n):03}'
        import math
        y, x = math.floor(lat), math.floor(lon)
        dt = f'Copernicus_DSM_COG_10_{ns(y)}_00_{ew(x)}_00_DEM'
        wc = ns(math.floor(lat / 3)*3) + ew(math.floor(lon / 3)*3)
        coords = cell['geometry_wgs84']['coordinates'][0]
        assert all(x <= p[0] <= x+1 and y <= p[1] <= y+1 for p in coords), 'cell requires tile mosaic'
        sample = {'region': region, 'intended_stratum': stratum, 'anchor_lonlat': [lon, lat], 'grid_cell': cell}
        for key, url, categorical in [
            ('elevation', f'https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{dt}/{dt}.tif', False),
            ('land_cover', f'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{wc}_Map.tif', True)]:
            try:
                sample[key] = static.summarize(url, cell['bounds_m'], categorical)
            except Exception as exc:
                sample[key] = {'url': url, 'error': str(exc)}
        new['samples'].append(sample)
        print('Static complete:', region, stratum, flush=True)
    (HERE / 'extended_static_manifest.json').write_text(json.dumps(new, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
