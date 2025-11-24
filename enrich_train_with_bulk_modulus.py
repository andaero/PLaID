from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from tqdm import tqdm

from cond_gen.alex_mp_analysis import get_actual_sg


def _composition_key_from_elements_str(elements_str: str) -> tuple[str, ...] | None:
    """
    Turn the stringified elements list from train.csv into a sorted tuple key.

    Example input: "['Co', 'Mn', 'Na', 'Ni', 'O']"
    Output: ('Co', 'Mn', 'Na', 'Ni', 'O')
    """
    try:
        elems = ast.literal_eval(elements_str)
        if not isinstance(elems, (list, tuple)):
            return None
        return tuple(sorted(str(e) for e in elems))
    except Exception:
        return None


def _composition_key_from_structure(struct: Structure) -> tuple[str, ...]:
    """
    Build a composition key from a pymatgen Structure.
    We use the set of element symbols, sorted, to be consistent with train.csv.
    """
    return tuple(sorted(el.symbol for el in struct.composition.elements))


def enrich_train_with_bulk_modulus(
    train_csv: Path,
    matbench_json: Path,
    output_csv: Path,
) -> None:
    """
    For each structure in the matbench log(K_VRH) set, find matching structures
    in the train set via StructureMatcher and copy over the (non-log) bulk
    modulus into a new column in the train dataframe.
    """

    # --- Load data ---
    print(f"Loading train data from {train_csv} ...")
    # The first column in train.csv is a pandas index; drop it by using index_col=0
    df_train = pd.read_csv(train_csv, index_col=0)

    print(f"Loading matbench data from {matbench_json} ...")
    df_kvrh = pd.read_json(matbench_json, orient="split")

    # --- Build composition keys for train set, grouped by key ---
    print("Building composition keys for train set ...")
    df_train["comp_key"] = df_train["elements"].map(_composition_key_from_elements_str)

    comp_to_train_indices: dict[tuple[str, ...], list[int]] = {}
    for idx, key in df_train["comp_key"].items():
        if key is None:
            continue
        comp_to_train_indices.setdefault(key, []).append(idx)

    print(f"Found {len(comp_to_train_indices)} unique composition keys in train set.")

    # --- Convert matbench structures and build their composition keys ---
    print("Converting matbench structures to pymatgen Structures ...")
    matbench_structs: list[Structure] = []
    matbench_keys: list[tuple[str, ...]] = []
    for s_dict in df_kvrh["structure"]:
        struct = Structure.from_dict(s_dict)
        matbench_structs.append(struct)
        matbench_keys.append(_composition_key_from_structure(struct))

    # --- Simple composition-level overlap statistics ---
    matbench_key_set = set(matbench_keys)
    train_key_set = set(comp_to_train_indices.keys())
    overlapping_keys = matbench_key_set & train_key_set

    num_matbench_with_train_match = sum(
        1 for key in matbench_keys if key in train_key_set
    )
    print(
        f"Matbench structures with at least one train composition match: "
        f"{num_matbench_with_train_match} / {len(matbench_structs)}"
    )
    print(
        f"Number of overlapping composition keys (train ∩ matbench): "
        f"{len(overlapping_keys)}"
    )

    # --- Prepare matcher and train-structure cache ---
    matcher = StructureMatcher()

    # Cache: comp_key -> list[(train_index, Structure)]
    comp_to_train_structs: dict[tuple[str, ...], list[tuple[int, Structure]]] = {}

    # Column for bulk modulus (non-log); initialize with NaNs
    bulk_col_name = "dft_bulk_modulus"
    if bulk_col_name not in df_train.columns:
        df_train[bulk_col_name] = np.nan

    # Column to store RMSD of the selected best match for each train row
    rms_col_name = "bulk_modulus_match_rmsd"
    if rms_col_name not in df_train.columns:
        df_train[rms_col_name] = np.nan

    # Column to record dataset of origin
    origin_col_name = "origin"
    if origin_col_name not in df_train.columns:
        df_train[origin_col_name] = "mp_train"

    # Track the best (lowest) RMS match seen so far for each train index globally
    best_rms_for_train: dict[int, float] = {}

    # Track which matbench indices were successfully matched to at least one train row
    matched_matbench_indices: set[int] = set()

    print("Starting structure matching and bulk modulus transfer ...")

    for mb_idx, (mb_struct, comp_key, log_k) in enumerate(
        tqdm(
            zip(matbench_structs, matbench_keys, df_kvrh["log10(K_VRH)"]),
            total=len(matbench_structs),
            desc="Matching matbench→train",
        )
    ):
        # Skip if there is no train structure with the same composition
        if comp_key not in comp_to_train_indices:
            continue

        # Lazily build and cache train structures for this composition key
        if comp_key not in comp_to_train_structs:
            train_indices = comp_to_train_indices[comp_key]
            train_structs: list[tuple[int, Structure]] = []
            for tr_idx in train_indices:
                cif_str = df_train.at[tr_idx, "cif"]
                try:
                    tr_struct = Structure.from_str(cif_str, fmt="cif")
                except Exception:
                    continue
                train_structs.append((tr_idx, tr_struct))
            comp_to_train_structs[comp_key] = train_structs

        train_structs = comp_to_train_structs[comp_key]
        if not train_structs:
            continue

        # Convert log10(K_VRH) to K_VRH
        try:
            k_vrh = float(10.0 ** float(log_k))
        except Exception:
            continue

        # Match against all train structures with this composition,
        # but keep only the best (lowest-RMSD) match.
        best_tr_idx = None
        best_rms = None

        for tr_idx, tr_struct in train_structs:
            try:
                rms_result = matcher.get_rms_dist(mb_struct, tr_struct)
            except Exception:
                rms_result = None

            if rms_result is None:
                continue

            # rms_result is a tuple like (rms, max_dist, translation)
            rms_val, *_ = rms_result

            if best_rms is None or rms_val < best_rms:
                best_rms = rms_val
                best_tr_idx = tr_idx

        # If we found at least one match, assign bulk modulus to that single best match,
        # but only if it improves (lowers) the best RMS we have seen so far for that
        # train index. This ensures each train row gets the label from its closest
        # matching matbench structure globally.
        if best_tr_idx is not None and best_rms is not None:
            prev_best = best_rms_for_train.get(best_tr_idx)
            if prev_best is None or best_rms < prev_best:
                best_rms_for_train[best_tr_idx] = best_rms
                df_train.at[best_tr_idx, bulk_col_name] = k_vrh
                df_train.at[best_tr_idx, rms_col_name] = best_rms
                matched_matbench_indices.add(mb_idx)

    # --- Append unmatched matbench structures as new rows ---
    unmatched_indices = [
        i for i in range(len(matbench_structs)) if i not in matched_matbench_indices
    ]
    if unmatched_indices:
        print(
            f"Appending {len(unmatched_indices)} unmatched matbench structures "
            "as new rows to the enriched train dataset."
        )
        new_rows = []
        for i in unmatched_indices:
            mb_struct = matbench_structs[i]
            log_k = df_kvrh["log10(K_VRH)"].iloc[i]
            try:
                k_vrh = float(10.0 ** float(log_k))
            except Exception:
                k_vrh = np.nan

            try:
                cif_str = mb_struct.to(fmt="cif")
            except Exception:
                cif_str = None

            # Initialize row with all existing columns set to NaN
            row = {col: np.nan for col in df_train.columns}
            row["cif"] = cif_str
            row[bulk_col_name] = k_vrh
            row[origin_col_name] = "matbench_bulk_mod"

            # Populate space group for unmatched structures using get_actual_sg
            if cif_str is not None and "spacegroup.number" in row:
                try:
                    sg_number = get_actual_sg({"cif": cif_str})
                except Exception:
                    sg_number = np.nan
                row["spacegroup.number"] = sg_number

            # rms column remains NaN for unmatched rows
            new_rows.append(row)

        df_unmatched = pd.DataFrame(new_rows)
        df_train = pd.concat([df_train, df_unmatched], ignore_index=True)

    # --- Finalize and save ---
    # Drop helper column
    df_train = df_train.drop(columns=["comp_key"])

    print(f"Saving enriched train data to {output_csv} ...")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(output_csv)
    print("Done.")


def main() -> None:
    project_root = Path(__file__).resolve().parent

    # Default paths based on current repository layout
    default_train = project_root / "data" / "basic" / "train.csv"
    default_matbench = project_root / "data" / "basic" / "matbench_log_kvrh.json"
    default_output = project_root / "data" / "enriched" / "train_with_bulk_modulus.csv"

    enrich_train_with_bulk_modulus(
        train_csv=default_train,
        matbench_json=default_matbench,
        output_csv=default_output,
    )


def summarize_bulk_modulus(path: str | Path) -> None:
    df = pd.read_csv(path, index_col=0)  # first column is index
    total_rows = len(df)
    if "bulk_modulus_K_VRH" not in df.columns:
        print("Column 'bulk_modulus_K_VRH' not found.")
        return

    non_null = df["bulk_modulus_K_VRH"].notna().sum()
    print(f"Total rows: {total_rows}")
    print(f"Rows with bulk_modulus_K_VRH: {non_null}")
    print(f"Fraction with bulk modulus: {non_null / total_rows:.4f}")

    # print number of rows with origin "matbench_bulk_mod"
    print(f"Num matbench_bulk_mod rows: {len(df[df['origin'] == 'matbench_bulk_mod'])}")
    print(f"Num mp_train rows: {len(df[df['origin'] == 'mp_train'])}")

    # Average RMSD for rows that actually have a bulk modulus label
    rms_col = "bulk_modulus_match_rmsd"
    if rms_col in df.columns:
        mask = df["bulk_modulus_K_VRH"].notna() & df[rms_col].notna()
        if mask.any():
            avg_rmsd = df.loc[mask, rms_col].mean()
            print(f"Average RMSD of matched structures: {avg_rmsd:.6f}")
        else:
            print("No rows with both bulk modulus and RMSD populated.")
    else:
        print("Column 'bulk_modulus_match_rmsd' not found.")


if __name__ == "__main__":
    main()
