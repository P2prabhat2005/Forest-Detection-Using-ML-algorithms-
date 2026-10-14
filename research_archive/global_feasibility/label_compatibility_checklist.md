# Greece–British Columbia label-contract gate

**Proposed target:** For each supported ~25 km cell and local reporting day, indicate whether at least one *newly recorded, agency-confirmed, unplanned wildland fire* was assigned to that cell on its **recorded discovery day**. Any final size qualifies. This is discovery-day occurrence, not physical ignition forecasting. The forecast is issued before the next reporting day; the day boundary and issue-time rule must be frozen after both agencies confirm their calendars.

**Current verdict: FAIL / not yet demonstrated.** BC publicly describes `IGNITION_DATE` as discovery date, but its reporting calendar needs confirmation. The Greek `Ημερ/νία Έναρξης` / `Ώρα Έναρξης` meaning is unresolved. Do not construct positive or negative shared labels until the gate passes.

| Check | PASS only with evidence that… | FAIL if… | Now |
|---|---|---|---|
| Date semantics | Both fields are recorded discovery day, or equivalent source fields are obtainable | A field is estimated ignition/report/dispatch/opening with no comparable alternative | Pending |
| Day boundary | Agency calendar, time zone, DST and midnight handling are documented and compatible with issue time | Day cannot be assigned without ambiguous conversion | Pending |
| Event class | Confirmed unplanned wildland incidents can be selected consistently; no final-size cutoff | Prescribed, false, nuisance or non-wildland cases cannot be separated | Pending |
| Incident identity | One incident can be counted once despite repeats, complexes and corrections | Duplicate/merge rules cannot be resolved | Pending |
| Spatial assignment | Point meaning, CRS, accuracy and missing-point rule permit a defensible ~25 km cell | Point represents an unrelated response location or errors cannot be screened | Pending |
| Coverage / negatives | Covered geographies, dates, classes and finalization lag are documented | A zero may mean missing reporting, late entry or uncovered jurisdiction | Pending |
| Revision provenance | Stable IDs and correction/finalization policy permit reproducible as-of labels | Records may silently change without a reproducible snapshot rule | Pending |

## Evidence required before a shared dataset

1. **Agency-authored field definitions** for Greek start date/time and BC `IGNITION_DATE`, explicitly distinguishing discovery from ignition/report/dispatch/entry; version/date of those definitions.
2. **Reporting-calendar documentation** for both: local time zone(s), DST, midnight boundary, any historical changes, and whether date-only BC values are local-calendar dates. A few de-identified boundary examples if documentation is ambiguous.
3. **Authoritative classification codes and inclusion/exclusion rules** for confirmed unplanned wildland fires, including prescribed/escaped burns, false alarms, smoke chases and reclassifications; no ≥30 ha or other final-area qualification.
4. **Incident ID, point-location and correction rules:** stable unique key, duplicates/complexes/reactivations, coordinate meaning/CRS/accuracy, and handling of missing or revised locations.
5. **Coverage and finalization statements** by geography and period, including other-agency records, known omissions, reporting lag, backdating and archival revisions. Establish a source snapshot/version and cutoff before treating absence as a negative.

If any event lacks a defensible date, class or cell, label its cell/day **unknown**, not zero. Use zero only inside documented, finalized coverage after excluding known incomplete periods. If Greece cannot provide a discovery-day field, this proposed common target fails; renegotiate an actually shared endpoint before any cross-region evaluation.

**Source context:** [Greek public incident files](https://www.fireservice.gr/el/synola-dedomenon); [BC historical incident-point catalogue](https://catalogue.data.gov.bc.ca/dataset/bc-wildfire-fire-incident-locations-historical). Public metadata alone does not resolve the pending questions above.
