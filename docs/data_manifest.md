# Data manifest

The release run reads an immutable input bundle. Locally its root is `data/`; in
Code Ocean it is attached as the versioned Data Asset
`/data/epc_mirage_inputs`. `EPC_DATA_DIR` may point to another copy of the same
layout. Inputs are never modified during analysis.

Disposable derived frames, downloaded fallbacks and download checksum sidecars are
written to `EPC_SCRATCH_DIR` (`.cache/` locally and `/scratch` in Code Ocean). All
tables, figures and reports are written to `EPC_OUTPUT_DIR` (`outputs/` locally and
`/results` in Code Ocean). A failed transfer is kept under a temporary name and never
replaces a complete cache file. Existing downloads are verified against their
SHA-256 sidecars before reuse. Derived Parquet caches are also written atomically,
checksummed and paired with the digest of the immutable source that produced them;
a stale or altered scratch cache is rebuilt. Frozen external inputs atomically
replace any same-named scratch copy at process start.

## Input Data Asset layout

```text
epc_mirage_inputs/
  input_checksums.sha256
  nd_artefact_extended.parquet
  non_domestic_epc_unified_certificate_level.parquet  # Supplementary Section 1.18
  recommendations_long_clean.parquet                  # Supplementary Section 1.18
  france/
    dpe_window_2025H2_2026H1.parquet
    dpe_window_2025H2_2026H1.parquet.sha256
  external/
    gdp_deflator_hmt_jun2026.csv
    ipf_cost_schedule.csv
    sector_cost_crosswalk.csv
    subnat_elec_2005_2024.xlsx                         # optional source fallback
    subnat_gas_2005_2024.xlsx                          # optional source fallback
    metered_nondomestic_electricity_EW.csv
    metered_nondomestic_gas_EW.csv
    nd_need_2025_supporting_tables.xlsx
    nd_need_2024_geographical_annex.xlsx
    owid_carbon_intensity_electricity.csv
    ne_50m_admin_0_countries.geojson
    ne_50m_admin_0_map_subunits.geojson
```

`input_checksums.sha256` pins every supplied England/Wales, France and external snapshot.
The France parquet additionally requires an adjacent `.sha256` sidecar so the mounted
Data Asset remains independently verifiable. A checksum line uses the standard form
`<64-hex-digest>  <filename>`. The entry point verifies the manifest,
the 1,245,956-certificate/842,673-UPRN England/Wales contract, and the France schema
and manuscript row count before analysis. Filesystem aliases or text files containing
a path are not valid substitutes.

## England and Wales non-domestic EPC register

- Source: GOV.UK non-domestic EPC register, https://epc.opendatacommunities.org/
- Licence: Open Government Licence v3.0
- Preferred release file: `data/nd_artefact_extended.parquet`, a certificate-level
  table containing the analysis fields and extended robustness fields. The loader can
  instead construct it from `data/non_domestic_epc_unified_certificate_level.parquet`,
  a certificate-level table with, at minimum, the columns `certificate_number`,
  `uprn`, `inspection_date`, `asset_rating`, `building_emissions`,
  `standard_emissions`, `typical_emissions`, `primary_energy_value`, `floor_area`,
  `main_heating_fuel_clean`, `property_type_clean`, `transaction_type_clean`,
  `n_recommendations`, and the validity flags used by the loader.
- Derived subset: on first run the loader filters to valid asset ratings with a
  positive floor area and inspection years 2012 to 2025, tags the main-heating fuel
  group, and caches the analytic frame as
  `$EPC_SCRATCH_DIR/nd_artefact_analysis.parquet`
  (1,245,956 certificates across 842,673 unique UPRNs).
- Supplementary Section 1.18 additionally reads the wide unified register and
  `data/recommendations_long_clean.parquet`. Both are required by the default
  manuscript reproduction run and may be omitted only with `--skip-mae-rescue`.

## DESNZ subnational consumption statistics

- Source: DESNZ subnational electricity and gas consumption statistics, GOV.UK
- The release run requires the two frozen derived CSV series in `data/external/`
  so that Figure 2 uses the exact manuscript snapshot. Outside a publication run,
  the source workbooks can be supplied instead; a missing workbook is downloaded
  atomically to `$EPC_SCRATCH_DIR/external/` and checksummed.
- Used for the aggregate metered-energy comparison only. These are meter-level
  aggregates and are not linked to the assessed UPRNs.

## ND-NEED non-domestic building stock margins

- Source: DESNZ Non-domestic National Energy Efficiency Data Framework (ND-NEED),
  2025 supporting tables and the 2024 geographical annex, GOV.UK
- Read from the frozen `data/external/` snapshot when supplied. A missing snapshot is
  downloaded atomically to `$EPC_SCRATCH_DIR/external/` and checksummed.
- Used for the Stage-2 external-composition sensitivity (sector, size and region
  margins). ND-NEED's unit of observation is the building, so it is used for
  composition weighting only and never for record-level reconciliation.

## Grid carbon intensity and boundaries

- Electricity carbon intensity: Ember via Our World in Data, frozen under
  `data/external/`. Used for the structural-exposure typology.
- Country boundaries: Natural Earth 50m cultural vectors, frozen under
  `data/external/`.
  Map subunits are used so that England and Wales is drawn separately from Scotland
  and Northern Ireland.

## France DPE register

- Source: ADEME "DPE Logements existants (depuis juillet 2021)", dataset
  `dpe03existant`, https://data.ademe.fr/datasets/dpe03existant
- The release Data Asset contains the fixed window from 1 July 2025 to 30 June 2026
  at `data/france/dpe_window_2025H2_2026H1.parquet`. This frozen source is required
  for the manuscript run because the live ADEME dataset is mutable. Its record count,
  schema and SHA-256 digest are validated before analysis. Live downloads are an
  explicit data-refresh workflow and are written to scratch, not over this file.

## Cost inputs tracked in this repository

These three files are small, hand-assembled transcriptions of published sources and
are version-controlled so the expenditure-equivalent scale is fully traceable:

- `data/external/ipf_cost_schedule.csv` - archetype band-upgrade package costs from
  the Investment Property Forum commercial-building costing study (2017 prices).
- `data/external/gdp_deflator_hmt_jun2026.csv` - HM Treasury GDP deflator series used
  to rebase those costs to 2024-25 prices.
- `data/external/sector_cost_crosswalk.csv` - mapping from register property-type
  groups to the costing archetypes. Sectors without a defensible archetype are left
  unmatched rather than assigned a proxy cost.
