# EPC Mirage: grid decarbonisation can overstate retrofit progress in carbon-weighted building energy ratings

This repository contains the analysis code and input data for the study, which
examines the extent to which improvements in carbon-weighted building energy
ratings reflect revisions to the energy factors embedded in the rating metric
rather than changes in building performance. The analysis comprises three
components.

The first is the empirical core: an analysis of the England and Wales non-domestic
Energy Performance Certificate register, identified from the revision to the National
Calculation Methodology that took effect on 15 June 2022. Because that revision altered
the fuel-emission and primary-energy factors used in the calculation without altering
any building, it provides an accounting discontinuity against which certified
performance can be measured on a constant basis.

The second is a structural typology of national building-rating schemes across Europe,
establishing where the same mechanism is possible by construction, as a function of how
far each scheme's headline metric depends on revisable conversion coefficients.

The third is an external test in France, which recomputes existing residential DPE
certificates under the pre- and post-reform electricity primary-energy coefficients of
the 2026 reform. Holding every diagnostic input fixed, this isolates the effect of the
coefficient revision on the label itself.

## Citation

If you use this code or data, please cite the accompanying paper:

> Alex Melendez Ramos and Carles Vergara Alert. *EPC Mirage: grid decarbonisation
> can overstate retrofit progress in carbon-weighted building energy ratings.* 2026.

A BibTeX entry and DOI will be added on publication.

## System requirements

Python 3.11 or later is required; the pinned reference environment and
continuous-integration workflow use Python 3.13. The code runs on macOS and Linux,
needs no GPU or cluster, and can take up to a few hours end to end depending on
hardware. Direct package versions are pinned in `requirements.txt`; the fully
resolved environment is locked in `requirements-lock.txt`.

## Installation

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-lock.txt
    pip install --no-deps -e .

The lock includes `geopandas` and `pyogrio` for the map (Figure 4). For a lighter
install that omits the map, use `pip install -e .` and add it back later with
`pip install -e '.[map]'`.

## Usage

Run the complete analysis with the top-level driver:

    ./run

Every stochastic step is seeded, and the run first verifies the input checksums and
the frozen data contracts, so a missing or altered input fails immediately rather
than silently changing a result. Generated tables and figures are written under
`outputs/` (`outputs/tables/`, the main-text tables in `outputs/tables/paper/`, and
figures in `outputs/figures/`). `docs/output_manifest.md` maps every reported
quantity to the file and function that produce it.

Input, cache and output locations can be redirected without touching the code:

    EPC_DATA_DIR=/path/to/inputs \
    EPC_SCRATCH_DIR=/path/to/scratch \
    EPC_OUTPUT_DIR=/path/to/results \
    ./run

## Data

The complete input bundle is included under `data/`, with the large register and
register-derived tables stored through **Git LFS** (all `*.parquet` files). After
cloning, run `git lfs pull` to download them; to fetch only the code and small
inputs, clone with `GIT_LFS_SKIP_SMUDGE=1 git clone …`. The bundle contains the
cleaned England and Wales register, the frozen France window and all external source
snapshots, and is released under CC0 1.0 (public domain; see
[`data/LICENSE`](data/LICENSE)). Provenance, exact filenames, integrity checks and
the expected schemas are documented in `docs/data_manifest.md`.

The analysis never writes into `data/`; derived frames go to `.cache/` and results
to `outputs/`.

## Repository layout

    code/
      run                         analysis driver
      scripts/reproduce_all.py    single entry point for the whole pipeline
      src/epc_artefact/
        config.py           paths, the 15 June 2022 cut-off and the NCM factor constants
        data.py             register loading, cleaning and the cached analytic frame
        analysis.py         register trends, within-UPRN cohort, no-recorded-works
                            calibration, fixed-accounting bridge, decomposition
        markov.py           reassessment-throughput sensitivity and its validation
        robustness.py       certificate validity, England/Wales split, stricter no-works
        validation.py       held-out bridge validation and ND-NEED representativeness
        bridge_analysis.py  threshold model selection, transport, external-composition
                            sensitivity and the expenditure-equivalent scale
        dr_estimator.py     cross-fitted doubly robust threshold estimator
        mae_rescue.py       same-methodology reassessment noise floor
        typology.py         international structural-exposure coding
        map_figure.py       Figure 4
        figures.py          Figures 2, 3 and 5
        paper_tables.py     main-text Tables 1 to 3
        run.py              England and Wales stage
      doubly_robust_estimates/   reported doubly robust estimates
    run                           top-level analysis driver
    data/                        input bundle (large *.parquet via Git LFS) and external sources
    environment/                 Python 3.13 locked environment
    docs/                        data and output manifests
    tests/                       integrity, scientific and packaging checks
    .github/workflows/ci.yml     Linux Python 3.13 verification

## Reproducibility

Every stochastic step is seeded. The non-parametric bootstraps used for the reported
intervals draw from a single seed constant in `config.py`; their complete replicate
tables are written to the results directory. Cross-fitting folds are deterministic
partitions at UPRN level, and the France counterfactual is an exact same-certificate
transformation of the published register rather than a fitted model.

Every push and pull request is checked on Linux with Python 3.13: CI verifies that
the runtime and environment locks are identical, compiles the source, runs static
error checks, exercises the driver and runs the unit-test suite.

## License

The analysis code is released under the MIT License (see [`LICENSE`](LICENSE)). The
input data under `data/` is released under CC0 1.0 (see
[`data/LICENSE`](data/LICENSE)).
