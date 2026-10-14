# Draft — BC Wildfire Service data clarification (NOT SENT)

**To:** BCWILDFIREGEO@gov.bc.ca  
**Subject:** Clarification of historical wildfire discovery-day and incident records

Dear BC Wildfire Service data team,

We are assessing whether the public historical incident-point records can support a daily, area-based research target shared with Greece: **a newly recorded, agency-confirmed, unplanned wildland fire on its recorded discovery day**, regardless of eventual burned area. We are seeking definitions, not a bulk data extract. Could you please clarify:

1. The historical dataset documents `IGNITION_DATE` as **“the discovery date of the fire.”** Does it consistently mean first recorded discovery (not estimated physical ignition, first report, dispatch or database creation)? What calendar and time zone determine that date, including daylight-saving transitions and incidents near midnight? Is a more precise discovery timestamp retained, and did the convention change by year?
2. Does `FIRE_TYPE = Fire` reliably identify confirmed, unplanned wildland incidents? Which statuses/types must be excluded (for example prescribed fires, nuisance fires, smoke chases, field activities, training, duplicates)? How are escaped prescribed fires, suspected fires subsequently confirmed, and incidents later reclassified represented?
3. What do the published latitude/longitude represent—ignition/origin, first-discovery point, response location, or another incident point? What are the coordinate accuracy and rules for missing, rounded or later-corrected locations?
4. Which of `FIRE_ID`, `FIRE_NUMBER` and `FIRE_LABEL` should identify one incident across updates? How are repeat reports, complexes/mergers, overwintering/reactivated fires and cross-boundary incidents handled?
5. What is the geographic and temporal coverage of the historical file, including fires recorded by other agencies or on private/municipal land? When are seasons finalized, and can discovery dates, classes or locations be corrected after transfer from the current-fire dataset?

A versioned schema/code list, coverage and revision notes, plus a few de-identified edge-case examples would suffice. We are especially seeking confirmation of the `IGNITION_DATE` reporting-calendar convention before assigning daily labels.

Kind regards,  
Forest Fire Early-Warning research team

**Public source referenced:** [BC Wildfire Fire Incident Locations – Historical](https://catalogue.data.gov.bc.ca/dataset/bc-wildfire-fire-incident-locations-historical).
