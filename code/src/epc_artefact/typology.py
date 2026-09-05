"""International structural-exposure typology (paper Figure 3 / map + Supplementary table).

A structural exposure typology, not a causal estimate: England & Wales supplies the
causal within-building evidence; this identifies where factor-driven reclassification is
structurally *possible*. Metric-sensitivity score (carbon/factor exposure of the headline
label): delivered/final energy = 0, primary energy = 1, dual/CO2-reported = 1.5,
carbon-weighted = 2, not classified = missing. Threshold status is a SEPARATE variable.
Grid decarbonisation = % reduction in electricity-generation carbon intensity 2015->2023
(Ember via Our World in Data). Exposure_j = MetricSensitivity_j x max(0, GridReduction_j),
used only as an ordinal ranking.

Produces (all reproducible from the OWID download):
    outputs/tables/international_exposure_typology.csv
    outputs/tables/international_exposure_typology.tex   (Supplementary Table)
    outputs/figures/figure_4.pdf/.png                    (Figure 4)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (OWID_CARBON_INTENSITY_URL, OWID_CARBON_INTENSITY_CSV,
                     TYPOLOGY_START_YEAR as Y0, TYPOLOGY_END_YEAR as Y1,
                     OUT_TABLES, OUT_FIGURES, ensure_dirs)
from .data import download_if_missing

SENS = {"Delivered / final energy": 0.0, "Primary energy": 1.0,
        "Dual / CO2 reported": 1.5, "Carbon-weighted": 2.0, "Not classified": np.nan}

EU27 = ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark",
        "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy",
        "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal",
        "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"]
ENERGY_COMMUNITY = ["Albania", "Bosnia and Herzegovina", "Kosovo", "Montenegro",
                    "North Macedonia", "Serbia", "Ukraine", "Moldova"]

# Individually-verified national codings (override the framework default).
# (segment, headline_metric, metric_class, threshold_status, threshold_detail, confidence, source, notes)
VERIFIED = {
    "United Kingdom": ("Non-domestic (England & Wales)", "SBEM CO2 asset rating",
        "Carbon-weighted", "current", "MEES EPC-E minimum (in force); EPC-B 2030 proposed",
        "high", "NCM 2021/SBEM; MEES SI 2015/962; this study",
        "Asset rating = 50*BER/SER, a pure fuel-carbon ratio; the demonstrated case."),
    "France": ("Residential (DPE)", "Primary energy + GES (double threshold)",
        "Dual / CO2 reported", "current", "Passoire F/G letting restrictions (phased, in force)",
        "high", "Arrete 31 Mar 2021 & 13 Aug 2025; BPIE; this study",
        "Label = worse of primary-energy and GES class; 2026 reform cut the electricity PE coefficient 2.3->1.9."),
    "Spain": ("Residential / non-domestic (CEE)", "Non-renewable primary energy + CO2 (dual scale)",
        "Dual / CO2 reported", "planned", "EPBD MEPS (transposition pending)",
        "medium", "RD 390/2021; BPIE; EPBD recast", "Two parallel rated scales (PE and CO2)."),
    "Ireland": ("Residential (BER)", "Primary energy kWh/m2/yr with CO2 indicator",
        "Dual / CO2 reported", "planned", "EPBD MEPS (transposition pending)",
        "medium", "SEAI DEAP; EPBD recast", "BER primary-energy headline with a CO2 indicator."),
    "Netherlands": ("Residential / non-domestic (NTA 8800)", "Fossil primary energy kWh/m2",
        "Primary energy", "current", "Office Label-C obligation (in force 2023)",
        "medium", "NTA 8800; Bouwbesluit office Label-C", "Primary-energy headline; CO2 not the threshold."),
    "Greece": ("Residential / non-domestic (PEA)", "Primary energy kWh/m2/yr vs reference building",
        "Primary energy", "planned", "EPBD MEPS (transposition pending)",
        "medium", "YPEKA/KENAK; EPBD recast", "Primary-energy headline, 9-class scale."),
    "Norway": ("Residential / non-domestic (Energimerke)", "Delivered energy (kWh) + heating mark",
        "Delivered / final energy", "none", "No EPC-label minimum (EEA, outside EPBD MEPS)",
        "high", "NVE energy labelling of buildings",
        "Headline grade assesses DELIVERED energy -> invariant to PE/carbon factor revision."),
    "Switzerland": ("Residential / non-domestic (GEAK)", "Overall energy efficiency + direct CO2",
        "Dual / CO2 reported", "none", "Voluntary in most cantons; no federal MEPS",
        "medium", "GEAK (geak.ch)", "Cantonal certificate rating overall efficiency AND direct CO2."),
    "European Union (27)": ("Bloc reference", "Primary energy (EPBD default)",
        "Primary energy", "planned", "EPBD recast 2024/1275 MEPS for worst-performing stock",
        "medium", "EPBD recast 2024/1275", "Bloc reference; recast adds GWP indicators from 2028."),
}
COLS = ["country", "segment", "headline_metric", "metric_class", "threshold_status",
        "threshold_detail", "confidence", "source", "notes"]
EUROPE_SCOPE = (EU27 + ["Iceland", "Liechtenstein", "Norway", "Switzerland", "United Kingdom",
                        "Andorra"] + ENERGY_COMMUNITY + ["European Union (27)"])


def _row(country: str):
    if country in VERIFIED:
        return (country,) + VERIFIED[country]
    if country in EU27:
        return (country, "Residential / non-domestic (national EPC)", "Primary energy (EPBD)",
                "Primary energy", "planned", "EPBD MEPS (transposition by 29 May 2026)",
                "medium", "EPBD recast 2024/1275; BPIE", "EPBD-default; not individually audited.")
    if country in ("Iceland", "Liechtenstein"):
        return (country, "Residential / non-domestic (national EPC)", "Primary energy (EPBD via EEA)",
                "Primary energy", "planned", "EPBD adopted via EEA Agreement",
                "low", "EEA Joint Committee; EPBD", "EEA member applying EPBD; EPBD-default coding.")
    if country in ENERGY_COMMUNITY:
        return (country, "Residential / non-domestic (national EPC)", "Primary energy (EPBD via Energy Community)",
                "Primary energy", "none", "EPC scheme exists; no audited MEPS",
                "low", "Energy Community Treaty (EPBD adoption)", "Energy Community party; EPBD-default coding.")
    return (country, "n/a", "Not audited", "Not classified", "unknown", "Unknown",
            "low", "-", "Outside EU/EEA/Energy Community EPBD frameworks; not classified.")


def load_owid() -> pd.DataFrame:
    """OWID carbon-intensity-of-electricity (gCO2/kWh), long format; cached to data/external/."""
    ensure_dirs()
    download_if_missing(OWID_CARBON_INTENSITY_URL, OWID_CARBON_INTENSITY_CSV)
    return pd.read_csv(OWID_CARBON_INTENSITY_CSV)


def grid_intensity(owid: pd.DataFrame, country: str, year: int) -> float:
    col = next(c for c in owid.columns if "intensity" in c.lower())
    s = owid[(owid.Entity == country) & (owid.Year == year)][col]
    return float(s.iloc[0]) if len(s) else np.nan


def build_typology() -> pd.DataFrame:
    """The full per-country typology with grid decline and exposure index."""
    owid = load_owid()
    df = pd.DataFrame([_row(c) for c in EUROPE_SCOPE], columns=COLS)
    df["metric_sensitivity"] = df.metric_class.map(SENS)
    df["grid_gCO2_2015"] = df.country.map(lambda c: grid_intensity(owid, c, Y0)).round(1)
    df["grid_gCO2_2023"] = df.country.map(lambda c: grid_intensity(owid, c, Y1)).round(1)
    df["raw_pct_reduction"] = (100 * (df.grid_gCO2_2015 - df.grid_gCO2_2023) / df.grid_gCO2_2015).round(1)
    df["decarb_nonneg"] = df.raw_pct_reduction.clip(lower=0)
    df["exposure_index"] = (df.metric_sensitivity * df.decarb_nonneg).round(1)
    df = df.sort_values("exposure_index", ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0, "exposure_rank", np.where(df.exposure_index.notna(),
                                           df.exposure_index.rank(ascending=False, method="min"), np.nan))
    return df


# ── Supplementary Table (LaTeX) ────────────────────────────────────────────────
_MC = {"Carbon-weighted": "Carbon-weighted", "Dual / CO2 reported": "Dual / CO$_2$",
       "Primary energy": "Primary energy", "Delivered / final energy": "Delivered energy",
       "Not classified": "Not classified"}
_TH = {"current": "Current", "planned": "Planned", "none": "None", "unknown": "Unknown"}


def _short_source(country: str, src: str) -> str:
    src = str(src)
    m = {"United Kingdom": "This study; NCM/MEES", "France": r"This study; arr\^et\'es DPE",
         "Spain": "RD 390/2021; BPIE", "Ireland": "SEAI DEAP", "Greece": "KENAK; EPBD",
         "Netherlands": "NTA 8800", "Norway": "NVE", "Switzerland": "GEAK",
         "European Union (27)": "EPBD 2024/1275", "Andorra": "--"}
    if country in m:
        return m[country]
    if "Energy Community" in src:
        return r"Energy Comm.\ (EPBD)"
    if "EEA" in src:
        return "EEA; EPBD"
    return "EPBD 2024/1275; BPIE"


def write_typology_latex(df: pd.DataFrame, path: Path) -> Path:
    def sens(x):
        return "--" if pd.isna(x) else (f"{x:.1f}" if x != int(x) else f"{int(x)}")

    def pct(x):
        return "--" if pd.isna(x) else f"{x:.1f}"
    rows = []
    for _, r in df.iterrows():
        disp = "EU-27 (ref.)" if r.country == "European Union (27)" else r.country
        rows.append(" & ".join([disp, _MC.get(r.metric_class, r.metric_class), sens(r.metric_sensitivity),
                                 pct(r.raw_pct_reduction), _TH.get(r.threshold_status, "Unknown"),
                                 pct(r.exposure_index), str(r.confidence).capitalize(),
                                 _short_source(r.country, r.source)]) + r" \\")
    body = "\n".join(rows)
    tex = (r"""\begin{table*}[t!]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.05}
\caption{\textbf{International structural-exposure typology: per-country coding, grid decarbonisation, exposure index, sources and confidence.} Metric-sensitivity score: delivered/final energy = 0, primary energy = 1, dual or CO$_2$-reported = 1.5, carbon-weighted = 2, not classified = missing. Grid decline is the percentage reduction in electricity-generation carbon intensity """ + f"{Y0}--{Y1}" + r""" (Ember via Our World in Data). Policy-threshold status is coded separately from metric sensitivity. Exposure index $=$ metric sensitivity $\times \max(0,\text{grid decline})$ and is an ordinal ranking only, not a predicted reclassification rate. EU-27 members default to the EPBD primary-energy headline with planned minimum standards (Directive (EU) 2024/1275); non-EU/EEA and Energy-Community entries are coded by framework obligation at lower confidence, and verified national exceptions are coded individually. Rows are ordered by exposure index. Only the United Kingdom carries demonstrated (not merely structural) reclassification.}
\label{tab:international_typology}
\resizebox{\textwidth}{!}{
\begin{tabular}{@{}llccccll@{}}
\toprule
\textbf{Country} & \textbf{Metric class} & \textbf{Sens.} & \textbf{Grid decl.\ (\%)} & \textbf{Threshold} & \textbf{Exposure} & \textbf{Conf.} & \textbf{Source} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
}
\end{table*}
""")
    path.write_text(tex)
    return path


def run_typology() -> pd.DataFrame:
    ensure_dirs()
    df = build_typology()
    df.to_csv(OUT_TABLES / "international_exposure_typology.csv", index=False)
    write_typology_latex(df, OUT_TABLES / "international_exposure_typology.tex")
    return df


if __name__ == "__main__":
    d = run_typology()
    print(d[["country", "metric_class", "metric_sensitivity", "raw_pct_reduction",
             "exposure_index", "confidence"]].head(12).to_string(index=False))
    print(f"\nwrote {OUT_TABLES/'international_exposure_typology.csv'} and .tex ({len(d)} rows)")
