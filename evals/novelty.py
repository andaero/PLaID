"""
This file depends on and heavily modifies code from Meta's flowllm repository, which is MIT-licensed.
The original license is preserved.
"""

from __future__ import annotations

import io
from argparse import ArgumentParser, Namespace
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from toolz import compose

from evals.novelty_utils.joblib_ import joblib_map
from evals.novelty_utils.novelty_utils import save_metrics_only_overwrite_newly_computed
from evals.novelty_utils.pandas_ import (
    filter_prerelaxed,
    filter_prerelaxed_LLM,
    get_intersection,
    maybe_get_missing_columns,
)

from evals.novelty_utils.pymatgen_ import (
    COLUMNS_COMPUTATIONS,
    get_chemsys,
    to_structure,
)
from evals.novelty_utils.tabular import VALID_TABULAR_DATASETS, get_tabular_dataset

trap = io.StringIO()


def get_matches(
    structure: Structure, alternatives: pd.Series, matcher: StructureMatcher
) -> tuple[list[int], list[float]]:
    with redirect_stdout(trap):
        structure = to_structure(structure)
    matches, rms_dists = [], []
    for ind, alt in alternatives.items():
        with redirect_stdout(trap):
            alt_structure = to_structure(alt)
        rms_dist = matcher.get_rms_dist(structure, alt_structure)
        if rms_dist is not None:
            rms_dist, *_ = rms_dist
            rms_dists.append(rms_dist)
            matches.append(ind)
    return matches, rms_dists


def main(args: Namespace) -> None:
    # Initialize tracking dataframe with model name
    tracking_df = pd.DataFrame(index=[0])
    tracking_df["model_name"] = args.model_name

    original_csv_path = None
    if args.json:
        df = pd.read_json(args.input)
    else:
        # find root directory of the project
        root = Path(__file__).resolve().parents[2]
        csv_path = root / "crystal-text-llm" / args.input
        original_csv_path = csv_path
        og_df = pd.read_csv(csv_path)
        # rename structure column to "structure"
        df = og_df.rename(columns={args.structure_column: "structure"})
        # preserve the space group column
        if "actual_spacegroup" in og_df.columns:
            df["actual_spacegroup"] = og_df["actual_spacegroup"]
    print(df.keys())
    df = maybe_get_missing_columns(df, COLUMNS_COMPUTATIONS)
    print("length og", len(df))
    tracking_df["original_length"] = len(df)

    if args.ehulls is None:
        # filter out high energy structures
        df = df[df[args.e_above_hull_column] <= args.e_above_hull_maximum]
    else:
        df_hull = pd.read_json(args.ehulls)
        df = pd.merge(
            df, df_hull, left_index=True, right_on="original_index", how="inner"
        )
        # TODO: FOR NOVELTY CALC RANDOM 1K
        # df = df.sample(n=1000, random_state=420)

        # filter out high energy structures
        print("len after join", len(df))
        # update tracking_df original length to match the filtered df
        tracking_df["original_length"] = len(df)
        df.iloc[0].to_csv("test1.csv")
        df = df[df[args.e_above_hull_column] <= args.e_above_hull_maximum]
    if args.json:
        df = filter_prerelaxed(
            df,
            args.num_structures,
            maximum_nary=args.maximum_nary,
            minimum_nary=args.minimum_nary - 1,
        )
    else:
        df = filter_prerelaxed_LLM(
            df,
            args.num_structures,
            maximum_nary=args.maximum_nary,
            minimum_nary=args.minimum_nary - 1,
        )

    path_json_sun_count = args.output.parent / args.json_sun_count
    save_metrics_only_overwrite_newly_computed(
        path_json_sun_count, {"num_stable": len(df)}
    )
    print("num_stable", len(df))
    tracking_df["num_stable"] = len(df)

    matcher = StructureMatcher()  # MatterGen Novelty settings
    # matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)  # CDVAE settings

    # uniqueness
    matches_rms_dists_s = joblib_map(
        lambda structure: get_matches(structure, df["structure"], matcher),
        df["structure"].array,
        n_jobs=-4,
        inner_max_num_threads=1,
        desc="Matching for uniqueness",
        total=len(df),
    )
    # place those lists into a dataframe
    records = []
    for j, (matches, rms_dists) in enumerate(matches_rms_dists_s):
        assert len(matches) == len(rms_dists)
        ind_self = df.index[j]

        if len(matches) == 0:
            record = {
                "uniq_match_ind-0": pd.NA,
                "uniq_rms_dist_to-0": float("nan"),
            }
        elif len(matches) == 1:
            if ind_self == matches[0]:
                record = {
                    "uniq_match_ind-0": pd.NA,
                    "uniq_rms_dist_to-0": float("nan"),
                }
            else:
                print(
                    f"did not match self! Matched {ind_self} to {matches[0]} with RMSD {rms_dists[0]}"
                )
                record = {
                    "uniq_match_ind-0": matches[0],
                    "uniq_rms_dist_to-0": rms_dists[0],
                }
        else:
            record = {}
            for i, (match, rms_dist) in enumerate(zip(matches, rms_dists)):
                if ind_self == match:
                    record[f"uniq_match_ind-{i}"] = pd.NA
                    record[f"uniq_rms_dist_to-{i}"] = float("nan")
                else:
                    record[f"uniq_match_ind-{i}"] = pd.NA if np.isnan(match) else match
                    record[f"uniq_rms_dist_to-{i}"] = rms_dist
        records.append(record)
    uniq_out = pd.DataFrame.from_records(records, index=df.index)
    uniq_match_cols = [col for col in uniq_out.columns if col.startswith("uniq_match")]
    uniq_out[uniq_match_cols] = uniq_out[uniq_match_cols].astype("Int64")

    # Determine uniqueness: A structure is unique if it has no non-self matches
    is_unique = uniq_out[uniq_match_cols].isna().all(axis=1)
    og_df["is_unique" + str(args.e_above_hull_maximum)] = is_unique
    print(og_df.head())
    # load tabular data to compare to
    tds = get_tabular_dataset(args.tabular_dataset)
    print(tds)
    # novelty
    outs = []
    print(tds.valid_stages)
    for stage in tds.valid_stages:
        if args.reprocess:
            tds.process(stage)
        tabular: pd.DataFrame = getattr(tds, stage + "_df")
        # compositions must match to compare the resulting structure
        gen_chemsys = df["composition"].map(compose(tuple, sorted, get_chemsys))
        tab_chemsys = tabular["composition"].map(compose(tuple, sorted, get_chemsys))
        intersection = get_intersection(gen_chemsys, tab_chemsys)
        gen_to_compare = df["structure"][gen_chemsys.isin(intersection)]
        tab_to_compare = tab_chemsys.isin(intersection)

        # now do pairwise comparisons between these filtered groups
        matches_rms_dists_s = joblib_map(
            lambda structure: get_matches(
                structure, tabular["cif"][tab_to_compare], matcher
            ),
            gen_to_compare.array,
            n_jobs=-2,
            inner_max_num_threads=1,
            desc="Matching for novelty",
            total=len(gen_to_compare),
        )

        # place those lists into a dataframe
        records = []
        for matches, rms_dists in matches_rms_dists_s:
            assert len(matches) == len(rms_dists)
            if len(matches) == 0:
                record = {
                    f"match_ind_{stage}-0": pd.NA,
                    f"rms_dist_to_{stage}-0": float("nan"),
                }
            else:
                record = {}
                for i, (match, rms_dist) in enumerate(zip(matches, rms_dists)):
                    record[f"match_ind_{stage}-{i}"] = match
                    record[f"rms_dist_to_{stage}-{i}"] = rms_dist
            records.append(record)
        out = pd.DataFrame.from_records(records, index=gen_to_compare.index)
        outs.append(out)
    out = pd.concat(outs, axis=1)
    out = pd.concat([uniq_out, out], axis=1)

    print(f"{len(df)=}")
    print(f"{len(out)=}")

    # Determine novelty: A structure is novel if it's not in the training set
    is_novel = out["match_ind_train-0"].isna()
    og_df["is_novel" + str(args.e_above_hull_maximum)] = (
        is_novel  # Add novelty column to the main dataframe
    )
    print(og_df.head())

    not_in_train = out[out["match_ind_train-0"].isna()]
    print(f"{len(not_in_train)=}")
    tracking_df["num_unique"] = is_unique.sum()

    # remove duplicates that are not in the training set
    has_a_generated_dupe = pd.concat(
        [
            ~not_in_train[col].isna()
            for col in not_in_train.columns
            if col.startswith("uniq_match")
        ],
        axis=1,
    ).any(axis=1)
    not_in_train_is_dupe = not_in_train[has_a_generated_dupe]
    # mark the duplicates, avoiding the first one that appears
    dupes = []
    cols = [col for col in not_in_train_is_dupe.columns if col.startswith("uniq_match")]
    for i, row in not_in_train_is_dupe[cols].iterrows():
        if i not in dupes:
            dupes.extend(row.array.dropna().tolist())

    sun_materials = not_in_train.drop(dupes, errors="ignore")
    print(f"{len(sun_materials)=}")
    tracking_df["num_sun_materials"] = len(sun_materials)

    save_metrics_only_overwrite_newly_computed(
        path_json_sun_count, {"num_sun_materials": len(sun_materials)}
    )

    out["sun"] = False
    out.loc[sun_materials.index, "sun"] = True

    print(tracking_df)
    # Save tracking dataframe to csv by appending it to existing sun.csv if it exists
    # Also merge the 'sun' column into the main df before returning
    og_df = og_df.join(out["sun"])  # Merge the 'sun' column

    if args.sg is not None:
        # check in og_df for sun column true AND space group equal to args.sg
        # verify num of sun materials is what is expected
        print(
            f"Total number of sun materials: {len(og_df[og_df['sun'].fillna(False)])}"
        )
        print(
            f"Number of rows with matching space group: {len(og_df[og_df['actual_spacegroup'] == args.sg])}"
        )
        sg_sun_materials = og_df[
            (og_df["sun"].fillna(False)) & (og_df["actual_spacegroup"] == args.sg)
        ]
        tracking_df["num_ssun"] = len(sg_sun_materials)
        tracking_df["num_correct_sg"] = len(
            og_df[og_df["actual_spacegroup"] == args.sg]
        )
        print(f"{len(sg_sun_materials)=}")

    # change sun column name to the hull threshold
    og_df = og_df.rename(columns={"sun": f"sun_{args.e_above_hull_maximum}"})

    if not args.no_out:
        if Path(args.sun_out).exists():
            tracking_df.to_csv(args.sun_out, mode="a", header=False)
        else:
            tracking_df.to_csv(args.sun_out)
    out.to_json(args.output)  # Keep saving the detailed output JSON

    # Return the modified DataFrame and the original CSV path
    return og_df, original_csv_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", type=Path, help="prerelaxed dataframe (json or csv)")
    parser.add_argument(
        "output", type=Path, help="output JSON dataframe with match details"
    )
    parser.add_argument("model_name", type=str, help="model name")
    parser.add_argument(
        "--json", action="store_true", help="input is json", default=False
    )
    parser.add_argument(
        "--tabular_dataset",
        type=str,
        choices=VALID_TABULAR_DATASETS,
        default="diffcsp_mp20",
    )
    parser.add_argument("-n", "--num_structures", type=int, default=None)
    parser.add_argument("--structure_column", type=str, default="relaxed_cif")
    parser.add_argument(
        "--maximum_nary",
        type=int,
        default=None,  # we know there aren't structures in the dataset with more than this
        help="Any queries to structures with higher nary are avoided.",
    )
    parser.add_argument(
        "--minimum_nary",
        type=int,
        default=2,
        help="Any queries to structures with lower nary are avoided.",
    )
    parser.add_argument("--ehulls", type=str, default=None)
    parser.add_argument("--e_above_hull_column", type=str, default="e_above_hull")
    parser.add_argument("--e_above_hull_maximum", type=float, default=0.0)
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument("--json_sun_count", type=str, default="sun_count.json")
    parser.add_argument("--sun_out", type=str, default="sun_eqv2.csv")
    parser.add_argument("--no_out", action="store_true")
    parser.add_argument("--both_e_hulls", action="store_true")

    # COND GEN
    parser.add_argument("--sg", type=int, default=None)

    args = parser.parse_args()

    modified_df, original_csv_path = main(args)

    # If the input was a CSV, save the modified dataframe back to the original path
    if original_csv_path:
        # Construct the new path with _sun appended before the extension
        original_dir = original_csv_path.parent
        original_stem = original_csv_path.stem
        original_suffix = original_csv_path.suffix
        new_csv_path = original_dir / f"{original_stem}_sun{original_suffix}"

        print(
            f"Saving updated dataframe with is_unique, is_novel, and sun columns to: {new_csv_path}"
        )
        # Ensure the directory exists before saving
        new_csv_path.parent.mkdir(parents=True, exist_ok=True)
        modified_df.to_csv(new_csv_path, index=False)
        print("Save complete.")

        if args.both_e_hulls:
            # Run for the other hull threshold
            if "sun_0.0" not in modified_df.columns:
                # Modify the argument value after parsing
                args.e_above_hull_maximum = 0.0  # Change to desired new value
                # For our tracking file, route to the stable file
                if "metastable" in args.sun_out:
                    args.sun_out = args.sun_out.replace("_metastable", "")
                args.input = new_csv_path

                print("Running for hull threshold 0.0")
                modified_df_new, _ = main(args)
                modified_df_new.to_csv(new_csv_path, index=False)
                print("Additional save complete.")
            elif "sun_0.1" not in modified_df.columns:
                # Modify the argument value after parsing
                args.e_above_hull_maximum = 0.1  # Change to desired new value
                # For our tracking file, route to the metastable file
                if "metastable" not in args.sun_out:
                    args.sun_out = args.sun_out.replace(".csv", "_metastable.csv")
                args.input = new_csv_path

                print("Running for hull threshold 0.1")
                modified_df_new, _ = main(args)  # run again
                modified_df_new.to_csv(new_csv_path, index=False)
                print("Additional save complete.")
