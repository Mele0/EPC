"""France DPE 2026 coefficient reform — factor-revision replication (paper Supplementary Section 2).

Same-certificate counterfactual: recompute every ADEME DPE certificate under the pre-reform
(2.3) and post-reform (1.9) electricity primary-energy coefficients, holding final energy,
GES emissions, surface and all diagnostic inputs fixed. Reproduces every France supplementary
table and the headline numbers.

    from epc_artefact.france import run_france
    run_france()            # download window if missing, then analyse + emit tables

Data: ADEME "DPE Logements existants (depuis juillet 2021)" (dataset id dpe03existant),
downloaded via the data-fair API for the +/-6 month window around 1 Jan 2026 and cached to
data/france/. Verified: the register already publishes primary energy at 1.9, so
EP(2.3) = EP(1.9) + 0.4 * electricity_final_energy exactly.

Official rules reconstructed from primary sources (see field-provenance / small-surface audits):
  * Electricity coefficient 2.3 -> 1.9: Arrete du 13 aout 2025 (JORFTEXT000052134589),
    modifying Annexe 3 of the Arrete du 31 mars 2021; in force for certificates established
    from 1 January 2026. Post-2026 certificates automatically use 1.9; pre-2026 certificates
    remain valid and may be updated by a free attestation.
  * Small-surface thresholds (dwellings < 40 m2 below 800 m altitude): Arrete du 25 mars 2024
    (JORF n0093, 20 April 2024), in force 1 July 2024. Class cut-points for both the primary
    energy (CEP, kWh/m2/yr) and greenhouse-gas (EGES, kgCO2/m2/yr) axes are tabulated per
    reference surface with linear interpolation; SS_CEP / SS_EGES below reproduce the annexe.
  * Label rule: final class = worse of the CEP and EGES class; the reform changes only the
    electricity CEP coefficient, so it can move the energy axis only.
"""
from __future__ import annotations

import json
import os
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import (DPE_API_BASE, FRANCE_CACHE_PARQUET, FRANCE_INPUT_PARQUET,
                     FRANCE_WINDOW_PARQUET,
                     FRANCE_WINDOW, DPE_COEF_OLD, DPE_COEF_NEW, OUT_TABLES,
                     ensure_dirs)
from .data import verify_checksum, verify_or_record_checksum, write_checksum_sidecar

COEF_GAP = DPE_COEF_OLD - DPE_COEF_NEW            # 0.4
REFORM_DATE = pd.Timestamp("2026-01-01")
FG = {"F", "G"}
LETTERS = ["A", "B", "C", "D", "E", "F", "G"]
QC_TOL_ABS, QC_TOL_REL = 1.0, 0.005
# Standard (>= 40 m2) residential DPE cut-points (upper bound of each class).
EP_BOUNDS = [(70, "A"), (110, "B"), (180, "C"), (250, "D"), (330, "E"), (420, "F"), (np.inf, "G")]
GES_BOUNDS = [(6, "A"), (11, "B"), (30, "C"), (50, "D"), (70, "E"), (100, "F"), (np.inf, "G")]

# ── Official small-surface thresholds (Arrete du 25 mars 2024, dwellings < 40 m2, < 800 m) ──
# Upper bound of classes A..F at each tabulated reference surface; linear interpolation between,
# clamped to the <=8 m2 row below 8 m2 and to the standard 40 m2 row at/above 40 m2.
SS_BREAKS = np.array([8., 9., 10., 15., 20., 25., 30., 35., 40.])
SS_CEP = {  # primary energy, kWh/m2/yr
    "A": [146, 134, 124, 100, 88, 81, 76, 73, 70],
    "B": [186, 174, 164, 140, 128, 121, 116, 113, 110],
    "C": [386, 355, 329, 263, 230, 210, 197, 188, 180],
    "D": [505, 464, 428, 333, 300, 280, 267, 258, 250],
    "E": [622, 574, 533, 421, 385, 363, 349, 338, 330],
    "F": [739, 685, 640, 514, 476, 454, 439, 428, 420]}
SS_EGES = {  # greenhouse gas, kgCO2/m2/yr
    "A": [11, 11, 10, 8, 8, 7, 7, 7, 6],
    "B": [16, 16, 15, 13, 13, 12, 12, 12, 11],
    "C": [44, 42, 40, 36, 34, 32, 32, 31, 30],
    "D": [68, 65, 62, 56, 54, 52, 52, 51, 50],
    "E": [90, 87, 84, 76, 74, 73, 72, 71, 70],
    "F": [122, 118, 115, 107, 104, 103, 102, 101, 100]}

FIELDS = ["numero_dpe", "date_etablissement_dpe", "date_reception_dpe", "etiquette_dpe",
          "etiquette_ges", "version_dpe", "methode_application_dpe", "type_batiment",
          "surface_habitable_logement", "conso_5_usages_ep", "conso_5_usages_par_m2_ep",
          "conso_5_usages_ef", "conso_5_usages_par_m2_ef", "emission_ges_5_usages",
          "emission_ges_5_usages_par_m2", "type_energie_n1", "conso_5_usages_ef_energie_n1",
          "type_energie_n2", "conso_5_usages_ef_energie_n2", "type_energie_n3",
          "conso_5_usages_ef_energie_n3", "type_energie_principale_chauffage",
          "type_energie_principale_ecs", "code_departement_ban", "code_region_ban"]
UA = {"User-Agent": "epc-research-france-dpe"}
MANUSCRIPT_WINDOW_RECORDS = 2_809_777
MANUSCRIPT_REPRODUCTION_PCT = 97.93
MANUSCRIPT_SUMMARY = {
    "n_valid": 2_734_987,
    "fg_2p3_n": 361_984,
    "fg_1p9_n": 258_899,
    "exits": 103_085,
    "exit_pct_passoires": 28.48,
    "exits_reproduced_only": 102_004,
    "electric_share_exits": 89.1,
    "entries": 0,
}
MANUSCRIPT_HEATING = {
    "Electricity": (1_122_161, 189_920, 91_836),
    "Gas": (1_065_224, 88_107, 6_818),
    "Wood/biomass": (89_460, 16_268, 2_006),
    "Heat network": (334_035, 8_633, 1_298),
    "Oil/other-fossil": (124_107, 59_056, 1_127),
}


class FranceValidationError(AssertionError):
    """Raised when a full France reproduction fails a scientific invariant."""


def france_cache_path(out: Path = FRANCE_WINDOW_PARQUET,
                      max_records: int | None = None) -> Path:
    """Return a cache path that cannot confuse a smoke sample with the full window."""
    out = Path(out)
    if max_records is None:
        return out
    if max_records <= 0:
        raise ValueError("max_records must be a positive integer")
    marker = f".sample-{max_records}"
    if out.stem.endswith(marker):
        return out
    return out.with_name(f"{out.stem}{marker}{out.suffix}")


# ── Download (ADEME data-fair API, resumable, streamed to parquet) ─────────────
def _fetch(url: str, retries: int = 5) -> dict:
    for k in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
                return json.load(r)
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError("unreachable")


def download_window(out: Path = FRANCE_WINDOW_PARQUET, window=FRANCE_WINDOW,
                    max_records: int | None = None,
                    expected_records: int | None = None,
                    expected_sha256: str | None = None) -> Path:
    """Download the requested window to an atomic, checksum-recorded parquet cache.

    Smoke-test downloads always receive a sample-specific filename.  For a release
    run, ``expected_records`` makes source drift fail before the multi-million-row
    transfer starts.  The completed parquet is row-count checked before it replaces
    any existing cache.
    """
    ensure_dirs()
    out = france_cache_path(out, max_records)
    out.parent.mkdir(parents=True, exist_ok=True)
    qs = f"date_etablissement_dpe:[{window[0]} TO {window[1]}]"
    url = DPE_API_BASE + "?" + urllib.parse.urlencode(
        {"size": 10000, "select": ",".join(FIELDS), "qs": qs, "sort": "numero_dpe"})
    total = _fetch(DPE_API_BASE + "?" + urllib.parse.urlencode({"size": 0, "qs": qs}))["total"]
    if expected_records is not None and total != expected_records:
        raise FranceValidationError(
            f"ADEME window contains {total:,} records, but this release expects "
            f"{expected_records:,}. Supply the frozen, checksummed release parquet "
            f"at {FRANCE_INPUT_PARQUET} instead of rebuilding it from the live API.")
    target = min(total, max_records) if max_records else total
    print(f"France DPE window {window[0]}..{window[1]}: {total:,} records; downloading {target:,}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{out.name}.", suffix=".part", dir=out.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    writer, got, page = None, 0, 0
    try:
        while url and got < target:
            d = _fetch(url)
            rows = d.get("results", [])
            if not rows:
                break
            rows = rows[:target - got]
            table = pa.Table.from_pylist([{k: (None if r.get(k) in ("", None) else r.get(k))
                                           for k in FIELDS} for r in rows])
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="snappy")
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
            got += len(rows)
            page += 1
            if page % 20 == 0:
                print(f"  {got:,}/{target:,}")
            url = d.get("next")
        if writer is not None:
            writer.close()
            writer = None
        if got != target:
            raise RuntimeError(
                f"Incomplete ADEME download: expected {target:,} rows, received {got:,}")
        if pq.ParquetFile(temporary).metadata.num_rows != target:
            raise RuntimeError(f"Parquet row-count verification failed for {temporary}")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        actual_sha256 = verify_checksum(temporary, expected_sha256)
        os.replace(temporary, out)
        write_checksum_sidecar(out, actual_sha256)
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    print(f"  done: {got:,} -> {out}")
    return out


# ── Label recompute + QC ──────────────────────────────────────────────────────
def _class(values, bounds):
    """Flat (>= 40 m2) class assignment from a list of (upper_bound, letter) cut-points."""
    cuts = [b for b, _ in bounds[:-1]]
    letters = np.array([l for _, l in bounds], dtype=object)
    out = letters[np.searchsorted(cuts, np.asarray(values, float), side="left")]
    out[~np.isfinite(np.asarray(values, float))] = None
    return out


def _ss_upper_bounds(surface, table):
    """(n, 6) surface-adjusted upper bounds of classes A..F. Missing surface -> standard
    (40 m2) thresholds; surface clamped to [8, 40] so <8 uses the smallest tabulated row
    and >=40 uses the standard thresholds."""
    s = np.asarray(surface, float)
    s = np.where(np.isnan(s), 40.0, np.clip(s, 8.0, 40.0))
    return np.column_stack([np.interp(s, SS_BREAKS, table[k]) for k in ["A", "B", "C", "D", "E", "F"]])


def _class_ss(values, surface, table):
    """Class assignment using surface-adjusted small-surface thresholds (exact official rule)."""
    x = np.asarray(values, float)
    bounds = _ss_upper_bounds(surface, table)
    idx = (x[:, None] > bounds).sum(axis=1)              # number of upper bounds exceeded -> 0..6
    out = np.array(LETTERS, dtype=object)[idx]
    out[~np.isfinite(x)] = None
    return out


def _ss_boundary(surface, table, upper_of):
    """Surface-adjusted single cut-point (upper bound of class ``upper_of``)."""
    s = np.asarray(surface, float)
    s = np.where(np.isnan(s), 40.0, np.clip(s, 8.0, 40.0))
    return np.interp(s, SS_BREAKS, table[upper_of])


def _worse(a, b):
    rank = {l: i for i, l in enumerate(LETTERS)}
    ra = np.array([rank.get(x, -1) for x in a]); rb = np.array([rank.get(x, -1) for x in b])
    return np.where(ra >= rb, a, b)


def _ascii_lower(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").map(
        lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().lower())


def _heat_group(s: pd.Series) -> pd.Series:
    sn = _ascii_lower(s)
    return pd.Series(np.select(
        [sn.str.contains("electricit"), sn.str.contains("gaz"),
         sn.str.contains("bois|granul|pellet"), sn.str.contains("reseau|urbain|chaleur"),
         sn.str.contains("fioul|charbon|gpl|propane|butane")],
        ["Electricity", "Gas", "Wood/biomass", "Heat network", "Oil/other-fossil"],
        default="Other"), index=s.index)


def build_frame(path: Path = FRANCE_WINDOW_PARQUET) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in ["surface_habitable_logement", "conso_5_usages_ep", "conso_5_usages_par_m2_ep",
              "conso_5_usages_ef", "emission_ges_5_usages_par_m2", "conso_5_usages_ef_energie_n1",
              "conso_5_usages_ef_energie_n2", "conso_5_usages_ef_energie_n3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date_etablissement_dpe"] = pd.to_datetime(df["date_etablissement_dpe"], errors="coerce")
    elec = np.zeros(len(df))
    for i in (1, 2, 3):
        e = df[f"conso_5_usages_ef_energie_n{i}"].fillna(0).to_numpy()
        elec += np.where(_ascii_lower(df[f"type_energie_n{i}"]).str.contains("electricit").to_numpy(), e, 0.0)
    df["electric_ef"] = elec
    df["ep_1p9"] = df["conso_5_usages_ep"]
    df["ep_2p3"] = df["conso_5_usages_ep"] + COEF_GAP * df["electric_ef"]
    ratio = np.where(df["conso_5_usages_ep"] > 0, df["ep_2p3"] / df["conso_5_usages_ep"], np.nan)
    df["ep_m2_1p9"] = df["conso_5_usages_par_m2_ep"]
    df["ep_m2_2p3"] = df["conso_5_usages_par_m2_ep"] * ratio
    df["surface"] = df["surface_habitable_logement"]
    df["small_lt40"] = df.surface < 40
    surf = df.surface.to_numpy()
    ges_m2 = df["emission_ges_5_usages_par_m2"].to_numpy()
    ep19 = df["ep_m2_1p9"].to_numpy(); ep23 = df["ep_m2_2p3"].to_numpy()

    # Surface-adjusted (exact official small-surface rule) — primary labels.
    df["ges_cls"] = _class_ss(ges_m2, surf, SS_EGES)
    df["en_cls_1p9"] = _class_ss(ep19, surf, SS_CEP)
    df["en_cls_2p3"] = _class_ss(ep23, surf, SS_CEP)
    df["label_1p9"] = _worse(df["en_cls_1p9"].to_numpy(), df["ges_cls"].to_numpy())
    df["label_2p3"] = _worse(df["en_cls_2p3"].to_numpy(), df["ges_cls"].to_numpy())

    # Flat (standard-threshold) recompute — kept only to quantify the small-surface correction.
    ges_flat = _class(ges_m2, GES_BOUNDS)
    df["ges_cls_flat"] = ges_flat
    df["label_1p9_flat"] = _worse(_class(ep19, EP_BOUNDS), ges_flat)
    df["label_2p3_flat"] = _worse(_class(ep23, EP_BOUNDS), ges_flat)

    # Distance to the surface-adjusted E|F energy cut-point (the boundary an exit crosses).
    ef_cut = _ss_boundary(surf, SS_CEP, "E")             # upper bound of E = E|F boundary
    df["ef_cut"] = ef_cut
    df["d_ef_1p9"] = np.abs(ep19 - ef_cut)
    df["d_ef_2p3"] = np.abs(ep23 - ef_cut)

    # QC guard
    ef = df["conso_5_usages_ef"]
    slot = df[[f"conso_5_usages_ef_energie_n{i}" for i in (1, 2, 3)]].fillna(0).sum(axis=1)
    tol = np.maximum(QC_TOL_ABS, QC_TOL_REL * ef)
    df["qc_elec_ok"] = (df.electric_ef >= 0) & (df.electric_ef <= ef + tol)
    df["qc_slot_ok"] = ef.notna() & ((slot - ef).abs() <= tol)
    df["qc_pass"] = df.qc_elec_ok & df.qc_slot_ok
    df["base_valid"] = (df["conso_5_usages_par_m2_ep"] > 0) & \
        df["emission_ges_5_usages_par_m2"].notna() & df["etiquette_dpe"].isin(LETTERS)
    df["valid"] = df.base_valid & df.qc_pass
    df["heat_group"] = _heat_group(df.type_energie_principale_chauffage)
    df["issued_2026"] = df.date_etablissement_dpe >= REFORM_DATE
    return df


def _surf_group(v: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(v.surface.isna(), "Missing surface",
                              np.where(v.surface < 40, "<40 m2", ">=40 m2")), index=v.index)


# ── Headline + supplementary results ──────────────────────────────────────────
def summarise(df: pd.DataFrame) -> dict:
    v = df[df.valid]
    old, new = v.label_2p3.isin(FG), v.label_1p9.isin(FG)
    exits = old & ~new
    base = df.base_valid
    return {
        "n_before_qc": int(base.sum()), "n_after_qc": int((base & df.qc_pass).sum()),
        "dropped": int((base & ~df.qc_pass).sum()),
        "n_valid": int(len(v)), "fg_2p3_n": int(old.sum()), "fg_1p9_n": int(new.sum()),
        "fg_2p3_pct": round(100 * old.mean(), 2), "fg_1p9_pct": round(100 * new.mean(), 2),
        "exits": int(exits.sum()), "exit_pct_passoires": round(100 * exits.sum() / max(old.sum(), 1), 2),
        "exit_pct_stock": round(100 * exits.mean(), 3), "entries": int(((~old) & new).sum()),
        # surface-adjusted (official small-surface rule) reproduction
        "match_all": round(100 * (v.label_1p9 == v.etiquette_dpe).mean(), 2),
        "match_ge40": round(100 * (v[v.surface >= 40].label_1p9 == v[v.surface >= 40].etiquette_dpe).mean(), 2),
        "match_lt40": round(100 * (v[v.small_lt40].label_1p9 == v[v.small_lt40].etiquette_dpe).mean(), 2),
        "match_missing": round(100 * (v[v.surface.isna()].label_1p9 == v[v.surface.isna()].etiquette_dpe).mean(), 2),
        # flat (standard-threshold) reproduction, for the small-surface-correction comparison
        "match_all_flat": round(100 * (v.label_1p9_flat == v.etiquette_dpe).mean(), 2),
        "match_lt40_flat": round(100 * (v[v.small_lt40].label_1p9_flat == v[v.small_lt40].etiquette_dpe).mean(), 2),
        # exits restricted to certificates whose published label we reproduce exactly
        "exits_reproduced_only": int((exits & (v.label_1p9 == v.etiquette_dpe)).sum()),
        "ge40_exits": int((exits & (v.surface >= 40)).sum()),
        "electric_share_exits": round(100 * (v.heat_group[exits] == "Electricity").mean(), 1),
    }


def transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    v = df[df.valid]
    return pd.crosstab(v.label_2p3, v.label_1p9).reindex(index=LETTERS, columns=LETTERS).fillna(0).astype(int)


# ── (1) Official small-surface threshold table + before/after reproduction ─────
def small_surface_thresholds(df: pd.DataFrame) -> dict:
    """Emit the official Arrete du 25 mars 2024 small-surface threshold table and the
    label-reproduction gain from implementing it (flat standard thresholds vs surface-
    adjusted thresholds), by surface group."""
    rows = []
    for j, s in enumerate(SS_BREAKS):
        row = {"reference_surface_m2": ("<=8" if s == 8 else f"{int(s)}")}
        for k in ["A", "B", "C", "D", "E", "F"]:
            row[f"CEP_{k}_upper"] = SS_CEP[k][j]
        for k in ["A", "B", "C", "D", "E", "F"]:
            row[f"EGES_{k}_upper"] = SS_EGES[k][j]
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_TABLES / "france_small_surface_thresholds.csv", index=False)

    v = df[df.valid]
    grp = _surf_group(v)
    cmp_rows = []
    for lab in ["<40 m2", ">=40 m2", "Missing surface", "All"]:
        m = pd.Series(True, index=v.index) if lab == "All" else (grp == lab)
        cmp_rows.append({
            "surface_group": lab, "n": int(m.sum()),
            "reproduction_flat_%": round(100 * (v.label_1p9_flat[m] == v.etiquette_dpe[m]).mean(), 2),
            "reproduction_smallsurface_%": round(100 * (v.label_1p9[m] == v.etiquette_dpe[m]).mean(), 2)})
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(OUT_TABLES / "france_small_surface_reproduction.csv", index=False)
    return {"thresholds": tab.to_dict("records"), "reproduction": cmp.to_dict("records")}


# ── (2) Exact label reproduction by surface x issue date + confusion matrix ────
def exact_label_reproduction(df: pd.DataFrame) -> dict:
    v = df[df.valid].copy()
    v["surf_group"] = _surf_group(v)
    v["date_group"] = np.where(v.issued_2026, "From 2026-01-01", "Before 2026-01-01")
    rows = []
    for sg in ["<40 m2", ">=40 m2", "Missing surface", "All"]:
        for dg in ["Before 2026-01-01", "From 2026-01-01", "All"]:
            m = pd.Series(True, index=v.index)
            if sg != "All":
                m &= v.surf_group == sg
            if dg != "All":
                m &= v.date_group == dg
            if m.sum() == 0:
                continue
            rows.append({"surface_group": sg, "issue_period": dg, "n": int(m.sum()),
                         "reproduction_%": round(100 * (v.label_1p9[m] == v.etiquette_dpe[m]).mean(), 2)})
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT_TABLES / "france_label_reproduction.csv", index=False)

    conf = pd.crosstab(v.etiquette_dpe, v.label_1p9).reindex(index=LETTERS, columns=LETTERS).fillna(0).astype(int)
    conf.to_csv(OUT_TABLES / "france_confusion_matrix.csv")
    mis = v[v.label_1p9 != v.etiquette_dpe]
    mm = (mis.groupby(["etiquette_dpe", "label_1p9"]).size().rename("n").reset_index()
          .rename(columns={"etiquette_dpe": "published_label", "label_1p9": "recomputed_label"})
          .sort_values("n", ascending=False))
    mm.to_csv(OUT_TABLES / "france_mismatch_by_label_pair.csv", index=False)
    return {"reproduction": rep.to_dict("records"),
            "n_mismatch": int(len(mis)),
            "mismatch_by_pair_top": mm.head(12).to_dict("records")}


# ── (3) Per-record mismatch audit + reproduced-label-only sensitivity ──────────
def label_mismatch_audit(df: pd.DataFrame) -> dict:
    v = df[df.valid].copy()
    mis = v[v.label_1p9 != v.etiquette_dpe].copy()
    # nearest CEP / EGES boundary distance (relative) under surface-adjusted thresholds
    cep_bounds = _ss_upper_bounds(mis.surface.to_numpy(), SS_CEP)
    eges_bounds = _ss_upper_bounds(mis.surface.to_numpy(), SS_EGES)
    ep = mis.ep_m2_1p9.to_numpy()[:, None]; ge = mis.emission_ges_5_usages_par_m2.to_numpy()[:, None]
    near_cep = (np.abs(ep - cep_bounds) / np.maximum(cep_bounds, 1e-9)).min(axis=1) <= 0.02
    near_eges = (np.abs(ge - eges_bounds) / np.maximum(eges_bounds, 1e-9)).min(axis=1) <= 0.02
    ges_mismatch = mis.ges_cls.to_numpy() != mis.etiquette_ges.to_numpy()
    reason = np.select(
        [mis.surface.isna().to_numpy(),
         ges_mismatch,
         near_cep | near_eges,
         mis.small_lt40.to_numpy()],
        ["Missing/inconsistent surface",
         "GHG (EGES) axis mismatch",
         "Rounding/margin to threshold",
         "Small-surface residual (<40 m2)"],
        default="Energy axis / unresolved")
    mis["mismatch_reason"] = reason
    audit = (mis.groupby("mismatch_reason").size().rename("n").reset_index())
    audit["pct_of_mismatch"] = (100 * audit.n / max(len(mis), 1)).round(2)
    audit["pct_of_valid"] = (100 * audit.n / max(len(v), 1)).round(3)
    audit = audit.sort_values("n", ascending=False)
    audit.to_csv(OUT_TABLES / "france_label_mismatch_audit.csv", index=False)

    # reproduced-label-only sensitivity: exits among certificates we reproduce exactly
    repro = v.label_1p9 == v.etiquette_dpe
    old = v.label_2p3.isin(FG); exits = old & ~v.label_1p9.isin(FG)
    r_old = old & repro; r_ex = exits & repro
    sens = {"n_reproduced": int(repro.sum()),
            "reproduced_share_%": round(100 * repro.mean(), 2),
            "passoires_reproduced": int(r_old.sum()),
            "exits_reproduced": int(r_ex.sum()),
            "exit_%_passoires_reproduced": round(100 * r_ex.sum() / max(r_old.sum(), 1), 2),
            "exits_full": int(exits.sum()),
            "exits_reproduced_share_of_full_%": round(100 * r_ex.sum() / max(int(exits.sum()), 1), 2)}
    json.dump(sens, open(OUT_TABLES / "france_reproduced_only_sensitivity.json", "w"), indent=2)
    return {"audit": audit.to_dict("records"), "reproduced_only": sens}


# ── (4) Issue-date audit (monthly) + coarse before/after split ─────────────────
def by_issue_date(df: pd.DataFrame) -> pd.DataFrame:
    """Coarse split: before vs from 1 January 2026 (reform effective date)."""
    v = df[df.valid].copy()
    old = v.label_2p3.isin(FG); exits = old & ~v.label_1p9.isin(FG)
    rows = []
    for lab, m in [("Issued before 2026-01-01", ~v.issued_2026.values),
                   ("Issued from 2026-01-01", v.issued_2026.values)]:
        o, e = old.values & m, exits.values & m
        match = (v.label_1p9.values[m] == v.etiquette_dpe.values[m]).mean()
        rows.append({"issue_period": lab, "n": int(m.sum()), "passoires": int(o.sum()),
                     "exits": int(e.sum()),
                     "exit_%_passoires": round(100 * e.sum() / max(o.sum(), 1), 2),
                     "published_label_reproduction_%": round(100 * match, 2)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "france_by_issue_date.csv", index=False)
    return out


def issue_date_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly issue-date bins with reproduction, exits and composition diagnostics."""
    v = df[df.valid].copy()
    v["month"] = v.date_etablissement_dpe.dt.to_period("M").astype(str)
    old = v.label_2p3.isin(FG); exits = old & ~v.label_1p9.isin(FG)
    v["_old"] = old.values; v["_exit"] = exits.values
    v["_repro"] = (v.label_1p9 == v.etiquette_dpe).values
    v["_elec"] = (v.heat_group == "Electricity").values
    rows = []
    for mth, s in v.groupby("month"):
        rows.append({
            "month": mth, "n": int(len(s)),
            "reproduction_%": round(100 * s._repro.mean(), 2),
            "passoires": int(s._old.sum()), "exits": int(s._exit.sum()),
            "exit_%_passoires": round(100 * s._exit.sum() / max(int(s._old.sum()), 1), 2),
            "share_electric_%": round(100 * s._elec.mean(), 1),
            "share_lt40_%": round(100 * (s.surface < 40).mean(), 1),
            "share_missing_surface_%": round(100 * s.surface.isna().mean(), 1)})
    out = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    out.to_csv(OUT_TABLES / "france_issue_date_audit.csv", index=False)
    return out


def field_provenance_audit(df: pd.DataFrame) -> dict:
    """Document which ADEME fields carry primary energy / provenance, and what is absent."""
    prov = {
        "primary_energy_field": "conso_5_usages_par_m2_ep (published at the 1.9 coefficient from 2026)",
        "electricity_final_field": "conso_5_usages_ef_energie_n{1,2,3} with type_energie_n{1,2,3}",
        "ghg_field": "emission_ges_5_usages_par_m2 (unaffected by the coefficient reform)",
        "issue_date_field": "date_etablissement_dpe",
        "method_version_field": "version_dpe",
        "version_dpe_distribution": {str(k): int(x) for k, x in
                                     df.version_dpe.value_counts(dropna=False).items()},
        "attestation_or_replacement_id_present": False,
        "stable_dwelling_identifier_present": False,
        "geography_available": "code_departement_ban, code_region_ban (no address / commune / geocode)",
        "note": ("The register exposes only the published primary-energy field; it does not flag "
                 "whether a pre-2026 certificate was updated by attestation, nor a stable dwelling "
                 "identifier. Provenance is therefore assessed indirectly via issue-date reproduction "
                 "rates rather than established directly.")}
    json.dump(prov, open(OUT_TABLES / "france_field_provenance.json", "w"), indent=2)
    return prov


# ── (5) Pre-2026 reconstruction diagnostic ─────────────────────────────────────
def pre2026_reconstruction(df: pd.DataFrame) -> pd.DataFrame:
    """For pre-2026 certificates, verify that recomputing the published 1.9 label from the
    exposed components reproduces the published label (by surface group and heating fuel),
    and that recomputed old-basis (2.3) F/G status is a strict superset of published F/G.
    The exact pre-2026 final-electricity split is the same n1/n2/n3 carrier fields used
    throughout; there is no separate archived 2.3 primary-energy field to compare against."""
    v = df[df.valid & ~df.issued_2026].copy()
    v["_repro"] = (v.label_1p9 == v.etiquette_dpe).values
    pub_fg = v.etiquette_dpe.isin(FG).values
    old_fg = v.label_2p3.isin(FG).values
    rows = []
    v["surf_group"] = _surf_group(v)
    for g, s in v.groupby("surf_group"):
        rows.append({"split": "surface", "group": g, "n": int(len(s)),
                     "reproduction_%": round(100 * s._repro.mean(), 2)})
    for g, s in v.groupby("heat_group"):
        rows.append({"split": "heating", "group": g, "n": int(len(s)),
                     "reproduction_%": round(100 * s._repro.mean(), 2)})
    out = pd.DataFrame(rows)
    # old-basis F/G must contain published F/G (reform only lowers energy => 2.3 >= 1.9 severity)
    contains = bool(np.all(old_fg[pub_fg])) if pub_fg.any() else True
    out.attrs["oldbasis_superset_of_published_fg"] = contains
    out.to_csv(OUT_TABLES / "france_pre2026_reconstruction.csv", index=False)
    return out


# ── (6) Binding-criterion audit ────────────────────────────────────────────────
def binding_criterion_audit(df: pd.DataFrame) -> dict:
    """Partition pre-reform (2.3) passoires by which axis binds, and confirm exits occur
    only where the primary-energy criterion is relaxed and the GES criterion is not binding."""
    v = df[df.valid]
    old = v.label_2p3.isin(FG)
    exits = old & ~v.label_1p9.isin(FG)
    en23 = v.en_cls_2p3.isin(FG)         # energy axis F/G under 2.3
    ges = v.ges_cls.isin(FG)             # GES axis F/G (surface-adjusted)
    part = pd.DataFrame({
        "binding_axis": ["Energy-limited (energy F/G, GES not)",
                         "GES-limited (GES F/G, energy not)",
                         "Both axes F/G"],
        "passoires": [int((old & en23 & ~ges).sum()),
                      int((old & ~en23 & ges).sum()),
                      int((old & en23 & ges).sum())],
        "exits": [int((exits & en23 & ~ges).sum()),
                  int((exits & ~en23 & ges).sum()),
                  int((exits & en23 & ges).sum())]})
    part["exit_%_of_row_passoires"] = (100 * part.exits / part.passoires.clip(lower=1)).round(2)
    part.to_csv(OUT_TABLES / "france_binding_criterion_audit.csv", index=False)
    summ = {"exits_total": int(exits.sum()),
            "exits_energy_limited": int((exits & ~ges).sum()),
            "exits_with_ges_binding": int((exits & ges).sum()),
            "passoires_staying_because_ges_binds": int((old & ges & ~exits).sum())}
    json.dump(summ, open(OUT_TABLES / "france_binding_criterion_summary.json", "w"), indent=2)
    return {"partition": part.to_dict("records"), "summary": summ}


# ── (7) Heating decomposition + negative controls ──────────────────────────────
def by_heating(df: pd.DataFrame) -> pd.DataFrame:
    v = df[df.valid].copy()
    g = v.groupby("heat_group")
    out = g.apply(lambda s: pd.Series({
        "n": len(s), "passoires": int(s.label_2p3.isin(FG).sum()),
        "exits": int((s.label_2p3.isin(FG) & ~s.label_1p9.isin(FG)).sum())}), include_groups=False).reset_index()
    out["exit_rate_passoires_%"] = (100 * out.exits / out.passoires.clip(lower=1)).round(2)
    out["share_of_exits_%"] = (100 * out.exits / out.exits.sum()).round(2)
    return out.sort_values("exits", ascending=False)


def negative_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Three falsification checks: (a) non-electric heating should exit far less than electric;
    (b) GES-binding passoires should essentially never exit; (c) passoires far from the E|F
    energy boundary should exit less than near-boundary passoires."""
    v = df[df.valid].copy()
    old = v.label_2p3.isin(FG); exits = old & ~v.label_1p9.isin(FG)
    elec = v.heat_group == "Electricity"; ges = v.ges_cls.isin(FG)
    near = v.d_ef_2p3 <= 20.0             # within 20 kWh/m2/yr of E|F boundary (2.3 basis)
    rows = []

    def add(label, mask):
        o = old & mask
        rows.append({"control": label, "passoires": int(o.sum()),
                     "exits": int((exits & mask).sum()),
                     "exit_%_passoires": round(100 * (exits & mask).sum() / max(int(o.sum()), 1), 2)})
    add("(a) Electric-heated passoires", elec)
    add("(a) Non-electric passoires (expect low)", ~elec)
    add("(b) GES not binding (can exit)", ~ges)
    add("(b) GES-binding passoires (expect ~0)", ges)
    add("(c) Near E|F boundary <=20 (expect high)", near)
    add("(c) Far from E|F boundary >20 (expect low)", ~near)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "france_negative_controls.csv", index=False)
    return out


# ── (8) Duplicate / repeated-dwelling sensitivity (proxy; no stable identifier) ─
def duplicate_sensitivity(df: pd.DataFrame) -> dict:
    """The ADEME open window exposes no stable dwelling identifier (only the per-certificate
    numero_dpe and department-level geography). We therefore report the certificate-flow
    result and a conservative proxy-deduplication that collapses certificates sharing
    department, rounded surface, rounded primary-energy and GES intensities, heating energy
    and published label, keeping the latest (and, as a check, one random) per proxy dwelling."""
    v = df[df.valid].copy()
    old = v.label_2p3.isin(FG); exits = old & ~v.label_1p9.isin(FG)
    v["_old"] = old.values; v["_exit"] = exits.values
    full = {"n_valid": int(len(v)), "passoires": int(old.sum()), "exits": int(exits.sum())}

    v["_proxy"] = (v.code_departement_ban.astype("string").fillna("NA") + "|"
                   + v.surface.round(0).astype("string").fillna("NA") + "|"
                   + v.ep_m2_1p9.round(0).astype("string").fillna("NA") + "|"
                   + v.emission_ges_5_usages_par_m2.round(0).astype("string").fillna("NA") + "|"
                   + v.heat_group.astype("string") + "|" + v.etiquette_dpe.astype("string"))
    n_unique = v._proxy.nunique()
    dup_share = round(100 * (1 - n_unique / len(v)), 2)
    rows = [{"basis": "Certificate flow (numero_dpe, all)", "n": int(len(v)),
             "passoires": int(old.sum()), "exits": int(exits.sum()),
             "exit_%_passoires": round(100 * exits.sum() / max(int(old.sum()), 1), 2)}]
    for lab, keep in [("Proxy-dedup (keep latest per proxy)", "last"),
                      ("Proxy-dedup (keep one random per proxy)", "random")]:
        if keep == "random":
            d = v.sample(frac=1.0, random_state=0).drop_duplicates("_proxy", keep="first")
        else:
            d = v.sort_values("date_etablissement_dpe").drop_duplicates("_proxy", keep="last")
        o = d._old; e = d._exit
        rows.append({"basis": lab, "n": int(len(d)), "passoires": int(o.sum()),
                     "exits": int(e.sum()),
                     "exit_%_passoires": round(100 * e.sum() / max(int(o.sum()), 1), 2)})
    # proxy-dedup restricted to >=40 m2
    d40 = v[v.surface >= 40].sort_values("date_etablissement_dpe").drop_duplicates("_proxy", keep="last")
    rows.append({"basis": "Proxy-dedup + >=40 m2", "n": int(len(d40)),
                 "passoires": int(d40._old.sum()), "exits": int(d40._exit.sum()),
                 "exit_%_passoires": round(100 * d40._exit.sum() / max(int(d40._old.sum()), 1), 2)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "france_duplicate_sensitivity.csv", index=False)
    # are exits concentrated in duplicated proxies?
    dup_mask = v._proxy.duplicated(keep=False)
    exit_dup_share = round(100 * (exits.values & dup_mask.values).sum() / max(int(exits.sum()), 1), 2)
    full.update({"proxy_unique": int(n_unique), "duplicate_share_%": dup_share,
                 "no_stable_identifier": True,
                 "exit_share_in_duplicated_proxies_%": exit_dup_share,
                 "table": out.to_dict("records")})
    json.dump(full, open(OUT_TABLES / "france_duplicate_sensitivity.json", "w"), indent=2, default=str)
    return full


# ── (9) Threshold-margin / rounding sensitivity ────────────────────────────────
def threshold_margin_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude passoires whose primary-energy intensity is within m kWh/m2/yr of the surface-
    adjusted E|F cut-point (on either the 1.9 or 2.3 basis) and recompute exits, to show the
    result is not a knife-edge rounding artefact. Also vary rounding of the intensity to the
    nearest integer before classifying."""
    v = df[df.valid].copy()
    old = v.label_2p3.isin(FG); exits = old & ~v.label_1p9.isin(FG)
    base_exits = int(exits.sum())
    rows = [{"rule": "No exclusion (full flow)", "excluded": 0, "passoires": int(old.sum()),
             "exits": base_exits, "exit_%_passoires": round(100 * base_exits / max(int(old.sum()), 1), 2),
             "exits_retained_%": 100.0}]
    dmin = np.minimum(v.d_ef_1p9.to_numpy(), v.d_ef_2p3.to_numpy())
    for m in (1, 2, 5, 10):
        keep = dmin > m
        o = old & keep; e = exits & keep
        rows.append({"rule": f"Exclude within {m} kWh/m2/yr of E|F", "excluded": int((~keep).sum()),
                     "passoires": int(o.sum()), "exits": int(e.sum()),
                     "exit_%_passoires": round(100 * e.sum() / max(int(o.sum()), 1), 2),
                     "exits_retained_%": round(100 * e.sum() / max(base_exits, 1), 2)})
    # rounding variation: round intensity to nearest integer before classifying
    surf = v.surface.to_numpy()
    l19r = _worse(_class_ss(np.round(v.ep_m2_1p9.to_numpy()), surf, SS_CEP),
                  _class_ss(np.round(v.emission_ges_5_usages_par_m2.to_numpy()), surf, SS_EGES))
    l23r = _worse(_class_ss(np.round(v.ep_m2_2p3.to_numpy()), surf, SS_CEP),
                  _class_ss(np.round(v.emission_ges_5_usages_par_m2.to_numpy()), surf, SS_EGES))
    oldr = pd.Series(l23r, index=v.index).isin(FG); newr = pd.Series(l19r, index=v.index).isin(FG)
    er = int((oldr & ~newr).sum())
    rows.append({"rule": "Round intensities to nearest integer", "excluded": 0,
                 "passoires": int(oldr.sum()), "exits": er,
                 "exit_%_passoires": round(100 * er / max(int(oldr.sum()), 1), 2),
                 "exits_retained_%": round(100 * er / max(base_exits, 1), 2)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "france_threshold_margin_sensitivity.csv", index=False)
    return out


# ── (10/11) Structural sensitivity table (main estimate hierarchy) ─────────────
def structural_sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    v = df[df.valid].copy()
    old = v.label_2p3.isin(FG); exits = old & ~v.label_1p9.isin(FG)
    base_exits = int(exits.sum())
    repro = v.label_1p9 == v.etiquette_dpe

    def rec(name, role, mask):
        o = old & mask; e = exits & mask
        return {"estimate": name, "role": role, "n": int(mask.sum()),
                "passoires": int(o.sum()), "exits": int(e.sum()),
                "exit_%_passoires": round(100 * e.sum() / max(int(o.sum()), 1), 2)}

    rows = [
        rec("Full certificate flow (surface-adjusted)", "MAIN", pd.Series(True, index=v.index)),
        rec("Reproduced-label-only certificates", "Robustness", repro),
        rec(">=40 m2 only", "Robustness", v.surface >= 40),
        rec("Excluding observed <40 m2", "Robustness", ~(v.surface < 40)),
    ]
    # windows
    for lab, lo, hi in [("+/-3m window", "2025-10-01", "2026-03-31"),
                        ("+/-1.5m window", "2025-11-15", "2026-02-15")]:
        rows.append(rec(lab, "Robustness",
                        (v.date_etablissement_dpe >= lo) & (v.date_etablissement_dpe <= hi)))
    # flat (pre-small-surface) method for reference
    oldf = v.label_2p3_flat.isin(FG); exf = oldf & ~v.label_1p9_flat.isin(FG)
    rows.append({"estimate": "Flat standard thresholds (superseded)", "role": "Reference",
                 "n": int(len(v)), "passoires": int(oldf.sum()), "exits": int(exf.sum()),
                 "exit_%_passoires": round(100 * exf.sum() / max(int(oldf.sum()), 1), 2)})
    # proxy-dedup headline (from duplicate_sensitivity, keep latest)
    v["_proxy"] = (v.code_departement_ban.astype("string").fillna("NA") + "|"
                   + v.surface.round(0).astype("string").fillna("NA") + "|"
                   + v.ep_m2_1p9.round(0).astype("string").fillna("NA") + "|"
                   + v.emission_ges_5_usages_par_m2.round(0).astype("string").fillna("NA") + "|"
                   + v.heat_group.astype("string") + "|" + v.etiquette_dpe.astype("string"))
    d = v.sort_values("date_etablissement_dpe").drop_duplicates("_proxy", keep="last")
    od = d.label_2p3.isin(FG); ed = od & ~d.label_1p9.isin(FG)
    rows.append({"estimate": "Proxy-deduplicated flow", "role": "Robustness", "n": int(len(d)),
                 "passoires": int(od.sum()), "exits": int(ed.sum()),
                 "exit_%_passoires": round(100 * ed.sum() / max(int(od.sum()), 1), 2)})
    out = pd.DataFrame(rows)
    out["share_of_main_exits_%"] = (100 * out.exits / max(base_exits, 1)).round(1)
    out.to_csv(OUT_TABLES / "france_structural_sensitivity.csv", index=False)
    return out


# ── Retained supplementary diagnostics ────────────────────────────────────────
def window_robustness(df: pd.DataFrame) -> pd.DataFrame:
    wins = [("+/-6.0m", "2025-07-01", "2026-06-30"), ("+/-3.0m", "2025-10-01", "2026-03-31"),
            ("+/-1.5m", "2025-11-15", "2026-02-15")]
    v0 = df[df.valid]; rows = []
    for lab, lo, hi in wins:
        d = v0[(v0.date_etablissement_dpe >= lo) & (v0.date_etablissement_dpe <= hi)]
        old, new = d.label_2p3.isin(FG), d.label_1p9.isin(FG); ex = old & ~new
        heat = d.heat_group
        rows.append({"window": lab, "start": lo, "end": hi, "n": int(len(d)),
                     "fg_2p3_%": round(100 * old.mean(), 2), "fg_1p9_%": round(100 * new.mean(), 2),
                     "exits": int(ex.sum()), "exit_%_stock": round(100 * ex.mean(), 3),
                     "exit_%_passoires": round(100 * ex.sum() / max(old.sum(), 1), 2),
                     "electric_%_exits": round(100 * (heat[ex] == "Electricity").mean(), 1)})
    return pd.DataFrame(rows)


def surface_robustness(df: pd.DataFrame) -> pd.DataFrame:
    v = df[df.valid]
    old = v.label_2p3.isin(FG)
    exits = old & ~v.label_1p9.isin(FG)
    n_exits = int(exits.sum())
    rows = []
    for lab, mask in [("Full QC-cleaned sample", pd.Series(True, index=v.index)),
                      ("Observed >=40 m2", v.surface >= 40),
                      ("Observed <40 m2", v.surface < 40),
                      ("Missing surface", v.surface.isna())]:
        o, e = old & mask, exits & mask
        rows.append({"subset": lab, "n": int(mask.sum()), "passoires": int(o.sum()),
                     "exits": int(e.sum()),
                     "exit_%_passoires": round(100 * e.sum() / max(o.sum(), 1), 2),
                     "share_of_full_exits_%": round(100 * e.sum() / max(n_exits, 1), 2)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "france_surface_robustness.csv", index=False)
    return out


def ges_binding(df: pd.DataFrame) -> dict:
    """With surface-adjusted GES thresholds, every exiter has a non-binding GES axis; the
    published-GES breakdown is reported alongside as a face-validity check."""
    v = df[df.valid]
    old = v.label_2p3.isin(FG)
    exits = old & ~v.label_1p9.isin(FG)
    binding = v.ges_cls.isin(FG)
    part = pd.DataFrame({
        "ges_status": ["GES A-E (energy-axis passoire, can exit)",
                       "GES F/G (emissions axis binding)"],
        "passoires": [int((old & ~binding).sum()), int((old & binding).sum())],
        "exits": [int((exits & ~binding).sum()), int((exits & binding).sum())]})
    part.to_csv(OUT_TABLES / "france_ges_binding.csv", index=False)
    pub = v.loc[exits, "etiquette_ges"].value_counts().reindex(list(LETTERS)).fillna(0).astype(int)
    pub_df = pd.DataFrame({"published_ges": pub.index, "exits": pub.values,
                           "pct_of_exits": (100 * pub.values / max(int(exits.sum()), 1)).round(2)})
    pub_df.to_csv(OUT_TABLES / "france_ges_published.csv", index=False)
    return {"binding": part.to_dict("records"), "published": pub_df.to_dict("records")}


def by_department(df: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    v = df[df.valid].copy()
    v["old"] = v.label_2p3.isin(FG)
    v["exit"] = v.old & ~v.label_1p9.isin(FG)
    v["elec"] = v.heat_group == "Electricity"
    v = v[v.code_departement_ban.notna()]
    tot_exits = int(v["exit"].sum())
    g = v.groupby("code_departement_ban")
    dep = g.agg(certificates=("old", "size"), passoires=("old", "sum"),
                exits=("exit", "sum")).reset_index().rename(columns={"code_departement_ban": "department"})
    dep["exit_rate_passoires_%"] = (100 * dep.exits / dep.passoires.clip(lower=1)).round(2)
    dep["share_of_exits_%"] = (100 * dep.exits / max(tot_exits, 1)).round(2)
    elec_share = v[v["exit"]].groupby("code_departement_ban").elec.mean() * 100
    dep["electric_exits_%"] = dep.department.map(elec_share).round(1)
    out = dep.sort_values("exits", ascending=False).head(top).reset_index(drop=True)
    out.to_csv(OUT_TABLES / "france_by_department.csv", index=False)
    return out


# ── Orchestration + sanity checks ─────────────────────────────────────────────
def validate_sanity_checks(checks: dict[str, bool]) -> None:
    """Raise one actionable error containing every failed France invariant."""
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    if failed:
        raise FranceValidationError(
            "France release validation failed: " + ", ".join(failed))


def manuscript_release_checks(total_records: int, summary: dict,
                              heating: pd.DataFrame) -> dict[str, bool]:
    """Compare a full France result with the frozen manuscript release contract."""
    checks = {
        "window_records_match_manuscript": bool(
            total_records == MANUSCRIPT_WINDOW_RECORDS),
        "repro_matches_manuscript": bool(
            abs(summary["match_all"] - MANUSCRIPT_REPRODUCTION_PCT) < 0.05),
    }
    for field, expected in MANUSCRIPT_SUMMARY.items():
        checks[f"headline_{field}_matches_manuscript"] = bool(summary[field] == expected)

    observed = heating.set_index("heat_group")[["n", "passoires", "exits"]]
    checks["heating_groups_match_manuscript"] = bool(
        set(observed.index) == set(MANUSCRIPT_HEATING))
    for group, expected in MANUSCRIPT_HEATING.items():
        present = group in observed.index
        actual = tuple(int(value) for value in observed.loc[group]) if present else None
        checks[f"heating_{group.lower().replace('/', '_').replace(' ', '_')}_matches_manuscript"] = bool(
            present and actual == expected)
    return checks


def sanity_checks(df: pd.DataFrame, res: dict, *, raise_on_failure: bool = False) -> dict:
    """Run scientific invariants and optionally make any failure fatal.

    Smoke samples retain the diagnostic booleans, while a full release run passes
    ``raise_on_failure=True`` and cannot continue with off-spec results.
    """
    v = df[df.valid]
    old = v.label_2p3.isin(FG); exits = old & ~v.label_1p9.isin(FG)
    heat = by_heating(df)
    surface_robustness(df)
    checks = manuscript_release_checks(len(df), res["summary"], heat)
    checks["exits_sum_by_heating"] = bool(int(heat.exits.sum()) == int(exits.sum()))
    # surface groups partition the valid sample exactly
    parts = int((v.surface >= 40).sum()) + int((v.surface < 40).sum()) + int(v.surface.isna().sum())
    checks["surface_groups_partition_valid"] = bool(parts == len(v))
    # surface-group exits sum to full exits
    seg = int((exits & (v.surface >= 40)).sum()) + int((exits & (v.surface < 40)).sum()) \
        + int((exits & v.surface.isna()).sum())
    checks["surface_exits_sum_to_full"] = bool(seg == int(exits.sum()))
    # reproduced-only subset of full
    repro = v.label_1p9 == v.etiquette_dpe
    checks["reproduced_subset_of_full"] = bool(int((exits & repro).sum()) <= int(exits.sum()))
    # issue-date split sums to full
    idsum = int((~v.issued_2026).sum()) + int(v.issued_2026.sum())
    checks["issue_date_partitions_valid"] = bool(idsum == len(v))
    # no entries into F/G
    checks["no_entries"] = bool(int((~old & v.label_1p9.isin(FG)).sum()) == 0)
    with (OUT_TABLES / "france_sanity_checks.json").open("w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2)
    if raise_on_failure:
        validate_sanity_checks(checks)
    return checks


def run_france(download: bool = True, max_records: int | None = None) -> dict:
    ensure_dirs()
    # Samples are disposable smoke-test inputs and must never be written beside
    # (or into the read-only mount containing) the immutable release snapshot.
    cache_root = FRANCE_CACHE_PARQUET if max_records is not None else FRANCE_WINDOW_PARQUET
    cache = france_cache_path(cache_root, max_records)
    if not cache.exists():
        if not download:
            raise FileNotFoundError(
                f"Missing France cache at {cache}; download is disabled")
        cache = download_window(
            out=cache,
            max_records=max_records,
            expected_records=(MANUSCRIPT_WINDOW_RECORDS if max_records is None else None))
    if cache == FRANCE_INPUT_PARQUET:
        verify_checksum(cache, require_checksum=True)
    else:
        verify_or_record_checksum(cache)
    df = build_frame(cache)
    summary = summarise(df)
    checks = sanity_checks(
        df, {"summary": summary}, raise_on_failure=max_records is None)
    res = {"summary": summary,
           "small_surface": small_surface_thresholds(df),
           "label_reproduction": exact_label_reproduction(df),
           "mismatch_audit": label_mismatch_audit(df),
           "field_provenance": field_provenance_audit(df),
           "pre2026_reconstruction": pre2026_reconstruction(df).to_dict("records"),
           "binding_criterion": binding_criterion_audit(df),
           "by_heating": by_heating(df).to_dict("records"),
           "negative_controls": negative_controls(df).to_dict("records"),
           "window_robustness": window_robustness(df).to_dict("records"),
           "surface_robustness": surface_robustness(df).to_dict("records"),
           "issue_date_audit": issue_date_audit(df).to_dict("records"),
           "duplicate_sensitivity": duplicate_sensitivity(df),
           "threshold_margin": threshold_margin_sensitivity(df).to_dict("records"),
           "structural_sensitivity": structural_sensitivity_table(df).to_dict("records"),
           "ges_binding": ges_binding(df),
           "by_department": by_department(df).to_dict("records"),
           "by_issue_date": by_issue_date(df).to_dict("records"),
           "sanity_checks": checks}
    transition_matrix(df).to_csv(OUT_TABLES / "france_transition_matrix.csv")
    by_heating(df).to_csv(OUT_TABLES / "france_by_heating.csv", index=False)
    window_robustness(df).to_csv(OUT_TABLES / "france_window_robustness.csv", index=False)
    with (OUT_TABLES / "france_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(res, handle, indent=2, default=str)
    return res


if __name__ == "__main__":
    import pprint
    pprint.pp(run_france(download=True)["summary"])
