"""Observe one future GFS 06Z run through the 11:00 UTC input cutoff.

Uses only small S3 .idx objects and NOMADS regional GRIB-filter cutouts. The
first successful client retrieval is observed availability, not publication.
No third-party Python packages are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from datetime import date, datetime, time as clock_time, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from extend_technical_audit import decode

HERE = Path(__file__).resolve().parent
TRACE_ROOT = HERE / "gfs_delivery_traces"
UTC = timezone.utc
REGIONS = {
    "Greece": (22.25, 39.25, 22.75, 39.75),
    "British_Columbia": (-120.75, 50.25, -120.25, 50.75),
}
FIELDS = {
    "TMP_2m": (0, 0, 103, 2, "K", "TMP", "2 m above ground"),
    "RH_2m": (1, 1, 103, 2, "%", "RH", "2 m above ground"),
    "UGRD_10m": (2, 2, 103, 10, "m/s", "UGRD", "10 m above ground"),
    "VGRD_10m": (2, 3, 103, 10, "m/s", "VGRD", "10 m above ground"),
    "APCP_surface": (1, 8, 1, 0, "kg/m2 = mm water", "APCP", "surface"),
}
INSTANT = tuple(k for k in FIELDS if k != "APCP_surface")
HEADERS = ("Date", "Last-Modified", "ETag", "Content-Length", "Content-Range", "Cache-Control", "Age")


def utc_at(day: date, hour: int) -> datetime:
    return datetime.combine(day, clock_time(hour), tzinfo=UTC)


def contract(day: date) -> dict:
    run = utc_at(day, 6)
    target_start = utc_at(day + timedelta(days=1), 0)
    target_end = target_start + timedelta(days=1)
    instant = [(target_start + timedelta(hours=h) - run) // timedelta(hours=1) for h in range(24)]
    rain = [(target_start + timedelta(hours=h + 1) - run) // timedelta(hours=1) for h in range(24)]
    return {
        "issue_utc": utc_at(day, 12).isoformat(), "cutoff_utc": utc_at(day, 11).isoformat(),
        "run_id": f"gfs.{day:%Y%m%d}/06", "run_initialization_utc": run.isoformat(),
        "target_start_utc": target_start.isoformat(), "target_end_exclusive_utc": target_end.isoformat(),
        "instantaneous_leads": instant, "precipitation_end_leads": rain,
        "all_leads": sorted(set(instant) | set(rain)),
        "precipitation_intervals": [
            {"start_utc": (target_start + timedelta(hours=h)).isoformat(),
             "end_utc": (target_start + timedelta(hours=h + 1)).isoformat(),
             "end_lead": rain[h], "block_start_lead": 6 * ((rain[h] - 1) // 6)}
            for h in range(24)
        ],
    }


def next_eligible(now: datetime) -> date:
    today = now.astimezone(UTC).date()
    return today + timedelta(days=1) if now >= utc_at(today, 11) else today


def headers_of(response) -> dict:
    return {key: response.headers.get(key) for key in HEADERS if response.headers.get(key) is not None}


def request_small(url: str, limit: int, log, kind: str, lead: int, region: str | None = None):
    begin = datetime.now(UTC)
    entry = {"kind": kind, "run_lead": lead, "region": region, "url": url, "started_utc": begin.isoformat()}
    body = None
    try:
        with urlopen(Request(url, headers={"User-Agent": "ForestFireFeasibilityTrace/1.0 (research)"}), timeout=25) as response:
            entry.update(http_status=response.status, headers=headers_of(response))
            body = response.read(limit + 1)
            if len(body) > limit:
                raise ValueError(f"response exceeds {limit} byte audit limit")
            entry.update(bytes=len(body),
                         sha256=hashlib.sha256(body).hexdigest())
    except HTTPError as exc:
        entry.update(http_status=exc.code, headers=headers_of(exc), error=str(exc))
    except (URLError, OSError, ValueError) as exc:
        entry.update(http_status=None, error=str(exc))
    entry["finished_utc"] = datetime.now(UTC).isoformat()
    entry["elapsed_seconds"] = round((datetime.fromisoformat(entry["finished_utc"]) - begin).total_seconds(), 3)
    log.write(json.dumps(entry) + "\n")
    log.flush()
    return body if entry.get("http_status") == 200 and "error" not in entry else None, entry


def index_url(day: date, lead: int) -> str:
    return f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{day:%Y%m%d}/06/atmos/gfs.t06z.pgrb2.0p25.f{lead:03}.idx"


def filter_url(day: date, lead: int, region: str, required: tuple[str, ...]) -> str:
    west, south, east, north = REGIONS[region]
    query = {"file": f"gfs.t06z.pgrb2.0p25.f{lead:03}", "dir": f"/gfs.{day:%Y%m%d}/06/atmos",
             "subregion": "", "leftlon": west, "rightlon": east, "bottomlat": south, "toplat": north}
    query.update({f"var_{FIELDS[name][5]}": "on" for name in required})
    if any(name in INSTANT for name in required):
        query.update({"lev_2_m_above_ground": "on", "lev_10_m_above_ground": "on"})
    if "APCP_surface" in required:
        query["lev_surface"] = "on"
    return "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?" + urlencode(query)


def required_fields(c: dict, lead: int) -> tuple[str, ...]:
    return ((INSTANT if lead in c["instantaneous_leads"] else ())
            + (("APCP_surface",) if lead in c["precipitation_end_leads"] else ()))


def check_index(body: bytes, c: dict, lead: int) -> dict:
    lines = body.decode("utf-8").splitlines()
    findings, errors = {}, []
    for name in required_fields(c, lead):
        _, _, _, _, unit, var, level = FIELDS[name]
        matches = [line for line in lines if f":d={c['run_id'][4:12]}06:{var}:{level}:" in line]
        if name == "APCP_surface":
            start = 6 * ((lead - 1) // 6)
            matches = [line for line in matches if f":{start}-{lead} hour acc fcst:" in line]
        else:
            matches = [line for line in matches if f":{lead} hour fcst:" in line]
        findings[name] = {"units": unit, "records": matches}
        if not matches:
            errors.append(f"{name} missing in index for f{lead:03}")
    return {"fields": findings, "errors": errors}


def check_cutout(body: bytes, c: dict, lead: int) -> dict:
    records = decode(body)
    findings, errors = {}, []
    run = datetime.fromisoformat(c["run_initialization_utc"])
    valid = run + timedelta(hours=lead)
    for name in required_fields(c, lead):
        cat, param, surface, value, unit, _, _ = FIELDS[name]
        matches = [r for r in records if (r["category"], r["parameter"], r["surface_type"], r["surface_value"])
                   == (cat, param, surface, value)]
        if name == "APCP_surface":
            start = 6 * ((lead - 1) // 6)
            matches = [r for r in matches if r["template"] == 8 and r["lead_start"] == start
                       and r["statistical_process"] == 1 and r["interval_unit"] == 1
                       and r["interval_hours"] == lead - start and r["end_utc"] == valid.isoformat()]
        else:
            matches = [r for r in matches if r["template"] == 0 and r["lead_start"] == lead]
        if not matches:
            errors.append(f"{name} missing or wrong timing in f{lead:03} cutout")
            continue
        first = matches[0]
        if any(r["values"] != first["values"] for r in matches[1:]):
            errors.append(f"conflicting duplicate {name} messages in f{lead:03}")
        if not first["values"] or any(not math.isfinite(v) for v in first["values"]):
            errors.append(f"missing/nonfinite {name} values in f{lead:03}")
        findings[name] = {"units": unit, "valid_utc": valid.isoformat(), "point_count": first["point_count"],
                          "missing_count": sum(not math.isfinite(v) for v in first["values"]),
                          "min": first["min"], "max": first["max"], "duplicate_messages": len(matches),
                          "grib_centre": first["centre"], "grib_master_table": first["master_table"],
                          "grib_local_table": first["local_table"], "grid_template": first["grid_template"],
                          "accumulation_start_utc": (run + timedelta(hours=first["lead_start"])).isoformat() if name == "APCP_surface" else None,
                          "accumulation_end_utc": first.get("end_utc") if name == "APCP_surface" else None,
                          "values": first["values"]}
    return {"fields": findings, "errors": errors}


def evaluate(c: dict, saved: dict) -> dict:
    missing = []
    first_times = []
    units = {}
    point_counts = {}
    rain = {}
    for region in REGIONS:
        for lead in c["all_leads"]:
            item = saved.get(region, {}).get(str(lead))
            for name in required_fields(c, lead):
                record = item.get("fields", {}).get(name) if item else None
                if record is None or item.get("errors"):
                    missing.append({"region": region, "lead": lead, "field": name,
                                    "reason": item.get("errors") if item else ["not retrieved before cutoff"]})
                    continue
                units[name] = record["units"]
                point_counts.setdefault(region, set()).add(record["point_count"])
                if record["missing_count"]:
                    missing.append({"region": region, "lead": lead, "field": name, "reason": "nonfinite values"})
                if name == "APCP_surface":
                    rain.setdefault(region, {})[lead] = record
            if item and item.get("first_success_utc"):
                first_times.append(item["first_success_utc"])
    rain_checks = {}
    for region, records in rain.items():
        intervals, increments, rain_errors = [], [], []
        for interval in c["precipitation_intervals"]:
            lead, start = interval["end_lead"], interval["block_start_lead"]
            current = records.get(lead)
            if current is None:
                rain_errors.append(f"missing f{lead:03}")
                continue
            if current["accumulation_start_utc"] != (datetime.fromisoformat(c["run_initialization_utc"]) + timedelta(hours=start)).isoformat():
                rain_errors.append(f"wrong accumulation start f{lead:03}")
            prior = records.get(lead - 1) if lead - 1 > start else None
            if lead - 1 > start and prior is None:
                rain_errors.append(f"missing baseline f{lead-1:03}")
                continue
            values = [v - u for v, u in zip(current["values"], prior["values"] if prior else [0.0] * len(current["values"]))]
            if any(v < -0.25 for v in values):
                rain_errors.append(f"negative hourly increment f{lead:03}")
            increments.append(values)
            intervals.append((interval["start_utc"], interval["end_utc"]))
        expected = [(x["start_utc"], x["end_utc"]) for x in c["precipitation_intervals"]]
        rain_checks[region] = {"hourly_intervals": len(intervals), "contiguous_without_overlap": intervals == expected,
                               "errors": rain_errors, "daily_mm_range": [min(map(sum, zip(*increments))), max(map(sum, zip(*increments)))] if len(increments) == 24 else None}
    complete = not missing and all(len(v) == 1 for v in point_counts.values()) and len(rain_checks) == len(REGIONS) and all(v["contiguous_without_overlap"] and not v["errors"] for v in rain_checks.values())
    return {"complete_by_cutoff": complete, "missing": missing, "precipitation_checks": rain_checks,
            "point_counts": {k: sorted(v) for k, v in point_counts.items()}, "units": units,
            "latest_observed_success_utc": max(first_times) if first_times else None,
            "first_publication_time_proven": False,
            "availability_definition": "first successful GET observed by this client; polling interval limits timestamp precision"}


def write_report(folder: Path, c: dict, saved: dict, state: str, started: str, finished: str):
    evaluation = evaluate(c, saved)
    compact = {"gate": "FAIL_PENDING_LABEL_COMPATIBILITY", "trace_state": state,
               "started_utc": started, "finished_utc": finished, "contract": c,
               "regions": {k: list(v) for k, v in REGIONS.items()}, "results": evaluation,
               "observations": saved, "request_log": "requests.jsonl",
               "publication_note": "Archive Last-Modified and run initialization are not first public publication; observed retrieval is client-specific."}
    (folder / "report.json").write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
    lines = [f"# GFS 06Z prospective delivery trace — {c['run_id']}", "",
             f"State: **{state}**. Overall Greece–BC gate: **FAIL pending label compatibility**.", "",
             f"Observed from {started} to {finished}; cutoff {c['cutoff_utc']}.",
             f"Target UTC interval: [{c['target_start_utc']}, {c['target_end_exclusive_utc']}).",
             f"Complete required fields by cutoff: **{evaluation['complete_by_cutoff']}**.",
             f"Missing/invalid field slots: **{len(evaluation['missing'])}**.", "",
             "Client-observed first retrievals, request times, response headers and hashes are in report.json and requests.jsonl.",
             "These observations do not establish the provider's first public publication time.", "",
             "Precipitation intervals are checked by within-block differencing and exact one-hour target coverage.",
             "See report.json for source units, all fields, missingness and accumulation intervals.", ""]
    for region, check in evaluation["precipitation_checks"].items():
        lines.append(f"- {region}: {check['hourly_intervals']} intervals; contiguous={check['contiguous_without_overlap']}; errors={len(check['errors'])}.")
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test():
    day = date(2026, 10, 12)
    c = contract(day)
    assert c["instantaneous_leads"] == list(range(18, 42))
    assert c["precipitation_end_leads"] == list(range(19, 43))
    assert len(c["precipitation_intervals"]) == 24
    assert all(datetime.fromisoformat(a["end_utc"]) - datetime.fromisoformat(a["start_utc"]) == timedelta(hours=1) for a in c["precipitation_intervals"])
    assert all(c["precipitation_intervals"][i]["end_utc"] == c["precipitation_intervals"][i + 1]["start_utc"] for i in range(23))
    assert next_eligible(utc_at(day, 11)) == day + timedelta(days=1)
    assert next_eligible(utc_at(day, 10)) == day
    # Decode one saved actual cutout; this is an offline parser check, not a prospective observation.
    sample = HERE / "audit_weather_cutouts" / "Greece_20261009_06_f042.grib2"
    if sample.exists():
        old = contract(date(2026, 10, 9))
        result = check_cutout(sample.read_bytes(), old, 42)
        assert not result["errors"] and result["fields"]["APCP_surface"]["point_count"] == 9
        archived = HERE / "extended_weather_manifest.json"
        if archived.exists():
            earlier = json.loads(archived.read_text(encoding="utf-8"))
            saved = {region: {} for region in REGIONS}
            for item in earlier["recent_cutouts"]:
                region = item["region"].replace(" ", "_")
                lead = item["lead_hours"]
                local = HERE / item["local_file"]
                checked = check_cutout(local.read_bytes(), old, lead)
                assert not checked["errors"], (region, lead, checked["errors"])
                saved[region][str(lead)] = {"first_success_utc": "2026-10-09T10:00:00+00:00", **checked}
            assert evaluate(old, saved)["complete_by_cutoff"]
    return {"preflight_utc": datetime.now(UTC).isoformat(), "state": "READY_NOT_RUN",
            "next_eligible_run": f"gfs.{next_eligible(datetime.now(UTC)):%Y%m%d}/06",
            "contract_example": c, "offline_parser_sample_checked": sample.exists(),
            "offline_complete_window_checked": sample.exists() and (HERE / "extended_weather_manifest.json").exists(),
            "global_gate": "FAIL_PENDING_LABEL_COMPATIBILITY"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, help="Future UTC issue date YYYY-MM-DD")
    parser.add_argument("--next-eligible", action="store_true", help="Choose today or tomorrow from the 11Z cutoff")
    parser.add_argument("--self-test", action="store_true", help="Offline preflight; makes no network requests")
    parser.add_argument("--interval-seconds", type=int, default=120, help="Poll interval, default 120 seconds")
    args = parser.parse_args()
    if args.self_test:
        outcome = self_test()
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = HERE / f"gfs_trace_preflight_{stamp}.json"
        path.write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
        print(f"Offline preflight passed; saved {path}")
        return 0
    if not 60 <= args.interval_seconds <= 900:
        parser.error("interval must be 60–900 seconds")
    chosen = args.date or next_eligible(datetime.now(UTC))
    c = contract(chosen)
    if datetime.now(UTC) >= utc_at(chosen, 11):
        parser.error("11:00 UTC cutoff has passed for this date; cannot call this prospective")
    folder = TRACE_ROOT / f"{chosen:%Y%m%d}_06Z_started_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    folder.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    saved, indexed = {region: {} for region in REGIONS}, {}
    print(f"Trace folder: {folder}\nRun: {c['run_id']} · target {c['target_start_utc']} to {c['target_end_exclusive_utc']} · cutoff {c['cutoff_utc']}", flush=True)
    # Wait for the future cycle without contacting an earlier run.
    while datetime.now(UTC) < utc_at(chosen, 6) - timedelta(minutes=5):
        remaining = (utc_at(chosen, 6) - timedelta(minutes=5) - datetime.now(UTC)).total_seconds()
        time.sleep(min(60, max(1, remaining)))
    state = "CUTOFF_REACHED"
    trace_error = None
    try:
        with (folder / "requests.jsonl").open("a", encoding="utf-8") as log:
            while datetime.now(UTC) < utc_at(chosen, 11):
                for lead in c["all_leads"]:
                    if datetime.now(UTC) >= utc_at(chosen, 11):
                        break
                    if lead not in indexed:
                        body, event = request_small(index_url(chosen, lead), 100_000, log, "s3_index", lead)
                        if body is not None and datetime.fromisoformat(event["finished_utc"]) < utc_at(chosen, 11):
                            check = check_index(body, c, lead)
                            if not check["errors"]:
                                indexed[lead] = {"first_success_utc": event["finished_utc"], "sha256": event["sha256"],
                                                 "headers": event["headers"], "fields": check["fields"]}
                                (folder / f"f{lead:03}.idx").write_bytes(body)
                            else:
                                log.write(json.dumps({"kind": "validation_failure", "lead": lead, "at_utc": datetime.now(UTC).isoformat(), "errors": check["errors"]}) + "\n")
                                log.flush()
                    if lead not in indexed:
                        continue
                    for region in REGIONS:
                        if str(lead) in saved[region] or datetime.now(UTC) >= utc_at(chosen, 11):
                            continue
                        body, event = request_small(filter_url(chosen, lead, region, required_fields(c, lead)),
                                                    64_000, log, "regional_grib_cutout", lead, region)
                        if body is None or datetime.fromisoformat(event["finished_utc"]) >= utc_at(chosen, 11):
                            continue
                        try:
                            check = check_cutout(body, c, lead)
                            if check["errors"]:
                                raise ValueError("; ".join(check["errors"]))
                            target = folder / f"{region}_f{lead:03}.grib2"
                            target.write_bytes(body)
                            saved[region][str(lead)] = {"first_success_utc": event["finished_utc"],
                                                        "index_first_success_utc": indexed[lead]["first_success_utc"],
                                                        "sha256": event["sha256"], "headers": event["headers"],
                                                        "local_file": target.name, "fields": check["fields"], "errors": []}
                        except (AssertionError, KeyError, ValueError) as exc:
                            log.write(json.dumps({"kind": "validation_failure", "lead": lead, "region": region,
                                                  "at_utc": datetime.now(UTC).isoformat(), "error": str(exc)}) + "\n")
                            log.flush()
                if all(len(saved[region]) == len(c["all_leads"]) for region in REGIONS):
                    state = "ALL_REQUIRED_FIELDS_OBSERVED"
                    break
                remaining = (utc_at(chosen, 11) - datetime.now(UTC)).total_seconds()
                if remaining > 0:
                    time.sleep(min(args.interval_seconds, remaining))
    except KeyboardInterrupt:
        state = "INTERRUPTED"
    except Exception as exc:
        state = "FAILED"
        trace_error = f"{type(exc).__name__}: {exc}"
        with (folder / "requests.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"kind": "runner_failure", "at_utc": datetime.now(UTC).isoformat(),
                                  "error": trace_error}) + "\n")
    finished = datetime.now(UTC).isoformat()
    write_report(folder, c, saved, state, started, finished)
    if trace_error:
        print(f"Runner failed: {trace_error}", file=sys.stderr)
    print(f"Saved {folder / 'report.md'} and report.json; complete={evaluate(c, saved)['complete_by_cutoff']}", flush=True)
    return 0 if evaluate(c, saved)["complete_by_cutoff"] else 1


if __name__ == "__main__":
    sys.exit(main())
