from argparse import ArgumentParser
import pandas as pd
from llm_finetune import get_crystal_string
from datasets import Dataset
import random
from utils import get_crystal_string_wyckoff_pyx
from tqdm.auto import tqdm
from llm_sample import condition_templates
from pathlib import Path
from cond_gen.sample_parse import get_real_space_group

# Suppress pymatgen warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")


def create_preference_dataset(input_path, output_path, threshold, cif_column):
    # Read the CSV input
    df = pd.read_csv(input_path)
    preference_data = []

    # Define prompt
    prompt = "Below is a description of a bulk material. Generate a description of the lengths and angles of the lattice vectors and then the element type and coordinates for each atom within the lattice:"

    # Loop through rows and create the preference data
    for _, row in df.iterrows():
        if float(row["e_above_hull"]) < threshold:
            crystal_str = get_crystal_string(row[cif_column])
            original_str = row["gen_str"].split("\n")[1:]
            original_str = "\n".join(original_str)

            temp = {"prompt": prompt, "chosen": crystal_str, "rejected": original_str}
            preference_data.append(temp)

            print(f"Chosen: {crystal_str}")
            print(f"Rejected: {original_str}")

    # Convert preference data to Hugging Face Dataset
    dataset = Dataset.from_list(preference_data)

    # Split the dataset into train/test splits (if needed)
    # Example: 80% train, 20% test split
    dataset = dataset.train_test_split(
        test_size=0.2, seed=42
    )  # Adjust the split ratio as needed

    # Save the dataset to disk in the format expected by Hugging Face
    dataset.save_to_disk(output_path)
    print(f"Preference dataset saved to {output_path}")


def create_preference_dataset_2(input_path, threshold, cif_column, args):
    # Read the CSV input
    df = pd.read_csv(input_path)
    preference_data = []

    # Define prompt
    prompt = "Below is a description of a bulk material. "
    if args.conditions == "e_above_hull":
        prompt += "The energy above the convex hull is 0. "
    prompt += (
        "Generate a description of the lengths and angles of the lattice vectors "
        "and then the element type and coordinates for each atom within the lattice:\n"
    )
    tqdm.pandas()

    # Apply the function to the dataframe
    def process_crystal_string(row):
        if float(row["e_above_hull"]) < threshold:
            if args.raw:
                return "\n".join(row["gen_str"].split("\n")[1:])
            else:
                return (
                    get_crystal_string_wyckoff_pyx(row[cif_column])
                    if args.wyckoff
                    else get_crystal_string(row[cif_column])
                )
        else:
            return "\n".join(row["gen_str"].split("\n")[1:])
        # return "\n".join(row["gen_str"].split("\n")[1:])

    df["processed_str"] = df.progress_apply(process_crystal_string, axis=1)

    # Separate accepts and rejects
    accepts = df.loc[df["e_above_hull"] < threshold, "processed_str"].tolist()
    rejects = df.loc[df["e_above_hull"] >= threshold, "processed_str"].tolist()
    for accept in accepts:
        # randomly sample 100 rejects
        rej_subset = random.sample(rejects, 3)
        for reject in rej_subset:
            temp = {"prompt": prompt, "chosen": accept, "rejected": reject}
            preference_data.append(temp)

    # Convert preference data to Hugging Face Dataset
    if preference_data:
        print(preference_data[0])
    return preference_data


def create_sg_preference_dataset(input_path, threshold, cif_column, args):
    # Get all CSV files in the input directory
    input_dir = Path(input_path)
    csv_files = list(input_dir.glob("*_eqv2_ehull_results.csv"))

    if not csv_files:
        raise ValueError(
            f"No CSV files found in {input_path} matching pattern '*_eqv2_ehull_results.csv'"
        )

    if not args.sg_14:
        # space groups we want
        sg_numbers = [1, 15, 38, 119, 143, 194, 216]
        # filter csv_files to only include sg_numbers
        csv_files = [f for f in csv_files if int(f.stem.split("_")[0]) in sg_numbers]
        print(f"filtered to {len(csv_files)} files")
    else:
        print(f"using all {len(csv_files)} files")
    all_preference_data = []

    # Process each file
    for csv_file in csv_files:
        # Extract space group number from filename
        sg_number = csv_file.stem.split("_")[0]

        # Read the CSV file
        df = pd.read_csv(csv_file)
        df = get_real_space_group(df, num_workers=None, strict=True)

        # Define space group specific prompt
        prompt = "Below is a description of a bulk material. "
        prompt += condition_templates["spacegroup_number"].format(
            spacegroup_number=sg_number
        )

        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "and then the element type and coordinates for each atom within the lattice:\n"
        )
        print(prompt)

        # Process crystal strings
        tqdm.pandas()

        # Helper to get the correct string
        def process_crystal_string(row):
            if args and args.raw:
                return "\n".join(row["gen_str"].split("\n")[1:])
            else:
                return (
                    get_crystal_string_wyckoff_pyx(row[cif_column])
                    if args and args.wyckoff
                    else get_crystal_string(row[cif_column])
                )

        df["processed_str"] = df.progress_apply(process_crystal_string, axis=1)

        # Classify by e_above_hull and space group matching
        stable_all_df = df[
            df["e_above_hull"] <= threshold
        ].copy()  # All structures with e_above_hull <= threshold
        unstable_df = df[
            (df["e_above_hull"] > threshold) | (df["e_above_hull"].isna())
        ].copy()  # also include invalid structures

        # Split stable materials into matching and non-matching space groups
        stable_matching_df = stable_all_df[
            stable_all_df["actual_spacegroup"] == int(sg_number)
        ].copy()
        stable_nonmatching_df = stable_all_df[
            stable_all_df["actual_spacegroup"] != int(sg_number)
        ].copy()

        print(f"\nProcessing space group {sg_number}:")
        print(
            f"Metastable (matching {sg_number}): {len(stable_matching_df)}, Metastable (non-matching): {len(stable_nonmatching_df)}"
        )
        print(f"Unstable: {len(unstable_df)}")

        # Create preference pairs
        # 1. Stable (matching SG) vs Stable (non-matching SG)
        for _, stable_matching_row in stable_matching_df.iterrows():
            for _ in range(args.ratio if args and hasattr(args, "ratio") else 2):
                if stable_nonmatching_df.empty:
                    break
                stable_nonmatching_row = stable_nonmatching_df.sample(1).iloc[0]
                all_preference_data.append(
                    {
                        "prompt": prompt,
                        "chosen": stable_matching_row["processed_str"],
                        "rejected": stable_nonmatching_row["processed_str"],
                    }
                )

        # 2. Stable vs Unstable
        for _, stable_row in stable_all_df.iterrows():
            for _ in range(args.ratio if args and hasattr(args, "ratio") else 2):
                if unstable_df.empty:
                    break
                unstable_row = unstable_df.sample(1).iloc[0]
                all_preference_data.append(
                    {
                        "prompt": prompt,
                        "chosen": stable_row["processed_str"],
                        "rejected": unstable_row["processed_str"],
                    }
                )
    if all_preference_data:
        print(all_preference_data[0])
        print(all_preference_data[-1])
    return all_preference_data


def create_sg_preference_dataset_novel(input_path, threshold, cif_column, args):
    # Get all CSV files in the input directory
    input_dir = Path(input_path)
    csv_files = list(input_dir.glob("*_eqv2_ehull_results_sun.csv"))

    if not csv_files:
        raise ValueError(
            f"No CSV files found in {input_path} matching pattern '*_eqv2_ehull_results_sun.csv'"
        )

    if not args.sg_14:
        # space groups we want
        sg_numbers = [1, 15, 38, 119, 143, 194, 216]
        # filter csv_files to only include sg_numbers
        csv_files = [f for f in csv_files if int(f.stem.split("_")[0]) in sg_numbers]
        print(f"filtered to {len(csv_files)} files")
    else:
        print(f"using all {len(csv_files)} files")
    all_preference_data = []

    # Process each file
    for csv_file in csv_files:
        # Extract space group number from filename
        sg_number = csv_file.stem.split("_")[0]

        # Read the CSV file
        df = pd.read_csv(csv_file)
        df = get_real_space_group(df, num_workers=None, strict=True)

        # Define space group specific prompt
        prompt = "Below is a description of a bulk material. "
        prompt += condition_templates["spacegroup_number"].format(
            spacegroup_number=sg_number
        )

        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "and then the element type and coordinates for each atom within the lattice:\n"
        )
        print(prompt)

        # Process crystal strings
        tqdm.pandas()

        # Helper to get the correct string
        def process_crystal_string(row):
            if args and args.raw:
                return "\n".join(row["gen_str"].split("\n")[1:])
            else:
                return (
                    get_crystal_string_wyckoff_pyx(row[cif_column])
                    if args and args.wyckoff
                    else get_crystal_string(row[cif_column])
                )

        df["processed_str"] = df.progress_apply(process_crystal_string, axis=1)

        # Classify by e_above_hull and space group matching
        stable_all_df = df[
            df["e_above_hull"] <= threshold
        ].copy()  # All structures with e_above_hull <= threshold
        unstable_df = df[
            (df["e_above_hull"] > threshold) | (df["e_above_hull"].isna())
        ].copy()  # also include invalid structures

        # Split stable materials into matching and non-matching space groups
        stable_matching_df = stable_all_df[
            stable_all_df["actual_spacegroup"] == int(sg_number)
        ].copy()
        stable_nonmatching_df = stable_all_df[
            stable_all_df["actual_spacegroup"] != int(sg_number)
        ].copy()
        novel_true_stable_df = stable_matching_df[
            stable_matching_df["is_novel0.1"] == True
        ].copy()
        novel_false_stable_df = stable_matching_df[
            stable_matching_df["is_novel0.1"] == False
        ].copy()

        print(f"\nProcessing space group {sg_number}:")
        print(
            f"Metastable (matching {sg_number}): {len(stable_matching_df)}, Metastable (non-matching): {len(stable_nonmatching_df)}"
        )
        print(f"Unstable: {len(unstable_df)}")

        # Create preference pairs
        # 1. Stable (matching SG) vs Stable (non-matching SG)
        for _, stable_matching_row in stable_matching_df.iterrows():
            for _ in range(args.ratio if args and hasattr(args, "ratio") else 2):
                if stable_nonmatching_df.empty:
                    break
                stable_nonmatching_row = stable_nonmatching_df.sample(1).iloc[0]
                all_preference_data.append(
                    {
                        "prompt": prompt,
                        "chosen": stable_matching_row["processed_str"],
                        "rejected": stable_nonmatching_row["processed_str"],
                    }
                )

        # 2. Stable Novel True Matching SG vs Stable Novel False Matching SG
        for _, novel_true_stable_row in novel_true_stable_df.iterrows():
            for _ in range(args.ratio if args and hasattr(args, "ratio") else 2):
                if novel_false_stable_df.empty:
                    break
                novel_false_stable_row = novel_false_stable_df.sample(1).iloc[0]
                all_preference_data.append(
                    {
                        "prompt": prompt,
                        "chosen": novel_true_stable_row["processed_str"],
                        "rejected": novel_false_stable_row["processed_str"],
                    }
                )

        # 2. Stable vs Unstable
        for _, stable_row in stable_all_df.iterrows():
            for _ in range(args.ratio if args and hasattr(args, "ratio") else 2):
                if unstable_df.empty:
                    break
                unstable_row = unstable_df.sample(1).iloc[0]
                all_preference_data.append(
                    {
                        "prompt": prompt,
                        "chosen": stable_row["processed_str"],
                        "rejected": unstable_row["processed_str"],
                    }
                )
    if all_preference_data:
        print(all_preference_data[0])
        print(all_preference_data[-1])
    return all_preference_data


def create_tiered_preference_dataset(
    input_path, threshold=0.08, cif_column="relaxed_cif", args=None
):
    df = pd.read_csv(input_path)
    preference_data = []

    # Define prompt
    prompt = "Below is a description of a bulk material. "
    if args and args.conditions == "e_above_hull":
        prompt += "The energy above the convex hull is 0. "
    prompt += (
        "Generate a description of the lengths and angles of the lattice vectors "
        "and then the element type and coordinates for each atom within the lattice:\n"
    )
    tqdm.pandas()

    # Helper to get the correct string
    def process_crystal_string(row):
        if args and args.raw:
            return "\n".join(row["gen_str"].split("\n")[1:])
        else:
            return (
                get_crystal_string_wyckoff_pyx(row[cif_column])
                if args and args.wyckoff
                else get_crystal_string(row[cif_column])
            )

    # Classify stability
    stable_df = df[df["e_above_hull"] <= 0].copy()
    metastable_df = df[
        (df["e_above_hull"] > 0) & (df["e_above_hull"] <= threshold)
    ].copy()
    unstable_df = df[df["e_above_hull"] > threshold].copy()

    print(
        f"Stable: {len(stable_df)}, Metastable: {len(metastable_df)}, Unstable: {len(unstable_df)}"
    )

    # Precompute processed strings
    stable_df["processed_str"] = stable_df.progress_apply(
        process_crystal_string, axis=1
    )
    metastable_df["processed_str"] = metastable_df.progress_apply(
        process_crystal_string, axis=1
    )
    unstable_df["processed_str"] = unstable_df.progress_apply(
        process_crystal_string, axis=1
    )

    # 1. Create (Stable, Metastable) pairs
    for _, stable_row in stable_df.iterrows():
        if len(metastable_df) == 0:
            break
        # Randomly sample one metastable per stable
        metastable_row = metastable_df.sample(1).iloc[0]
        temp = {
            "prompt": prompt,
            "chosen": stable_row["processed_str"],
            "rejected": metastable_row["processed_str"],
        }
        preference_data.append(temp)

        # 2. For every (stable, metastable), create two (stable, unstable) pairs
        for _ in range(2):
            if len(unstable_df) == 0:
                break
            unstable_row = unstable_df.sample(1).iloc[0]
            temp = {
                "prompt": prompt,
                "chosen": stable_row[
                    "processed_str"
                ],  # or you could also randomly choose stable/metastable here
                "rejected": unstable_row["processed_str"],
            }
            preference_data.append(temp)

    return preference_data


def create_tiered_preference_dataset_v2(
    input_path, output_path, threshold=0.08, cif_column="relaxed_cif", args=None
):
    df = pd.read_csv(input_path)
    preference_data = []

    # Define prompt
    prompt = "Below is a description of a bulk material. "
    if args and args.conditions == "e_above_hull":
        prompt += "The energy above the convex hull is 0. "
    prompt += (
        "Generate a description of the lengths and angles of the lattice vectors "
        "and then the element type and coordinates for each atom within the lattice:\n"
    )
    tqdm.pandas()

    # Helper to get the correct string
    def process_crystal_string(row):
        if args and args.raw:
            return "\n".join(row["gen_str"].split("\n")[1:])
        else:
            return (
                get_crystal_string_wyckoff_pyx(row[cif_column])
                if args and args.wyckoff
                else get_crystal_string(row[cif_column])
            )

    # Classify stability
    stable_df = df[df["e_above_hull"] <= 0].copy()
    metastable_df = df[
        (df["e_above_hull"] > 0) & (df["e_above_hull"] <= threshold)
    ].copy()
    unstable_df = df[df["e_above_hull"] > threshold].copy()

    print(
        f"Stable: {len(stable_df)}, Metastable: {len(metastable_df)}, Unstable: {len(unstable_df)}"
    )

    # Precompute processed strings
    stable_df["processed_str"] = stable_df.progress_apply(
        process_crystal_string, axis=1
    )
    metastable_df["processed_str"] = metastable_df.progress_apply(
        process_crystal_string, axis=1
    )
    unstable_df["processed_str"] = unstable_df.progress_apply(
        process_crystal_string, axis=1
    )

    # 1. Create (Stable, Metastable) pairs
    for _, stable_row in stable_df.iterrows():
        if len(metastable_df) == 0:
            break
        # Randomly sample one metastable per stable
        metastable_row = metastable_df.sample(1).iloc[0]
        temp = {
            "prompt": prompt,
            "chosen": stable_row["processed_str"],
            "rejected": metastable_row["processed_str"],
        }
        preference_data.append(temp)
        for _ in range(2):
            if len(unstable_df) == 0:
                break
            unstable_row = unstable_df.sample(1).iloc[0]
            temp = {
                "prompt": prompt,
                "chosen": stable_row["processed_str"],
                "rejected": unstable_row["processed_str"],
            }
            preference_data.append(temp)

    # 2. For every stable or metastable, create two (stable/metastable, unstable) pairs
    for _, row in metastable_df.iterrows():
        for _ in range(args.ratio):
            if len(unstable_df) == 0:
                break
            unstable_row = unstable_df.sample(1).iloc[0]
            temp = {
                "prompt": prompt,
                "chosen": row["processed_str"],
                "rejected": unstable_row["processed_str"],
            }
            preference_data.append(temp)

    return preference_data


def create_tiered_preference_dataset_v2_novel(
    input_path, output_path, threshold=0.08, cif_column="relaxed_cif", args=None
):
    df = pd.read_csv(input_path)
    preference_data = []

    # Define prompt
    prompt = "Below is a description of a bulk material. "
    if args and args.conditions == "e_above_hull":
        prompt += "The energy above the convex hull is 0. "
    prompt += (
        "Generate a description of the lengths and angles of the lattice vectors "
        "and then the element type and coordinates for each atom within the lattice:\n"
    )
    tqdm.pandas()

    # Helper to get the correct string
    def process_crystal_string(row):
        if args and args.raw:
            return "\n".join(row["gen_str"].split("\n")[1:])
        else:
            return (
                get_crystal_string_wyckoff_pyx(row[cif_column])
                if args and args.wyckoff
                else get_crystal_string(row[cif_column])
            )

    # Classify stability
    stable_df = df[df["e_above_hull"] <= 0].copy()
    metastable_df = df[
        (df["e_above_hull"] > 0) & (df["e_above_hull"] <= threshold)
    ].copy()
    unstable_df = df[
        (df["e_above_hull"] > threshold) | (df["e_above_hull"].isna())
    ].copy()

    print(
        f"Stable: {len(stable_df)}, Metastable: {len(metastable_df)}, Unstable: {len(unstable_df)}"
    )

    if "is_novel0.0" in stable_df.columns:
        novel_true_stable_df = stable_df[stable_df["is_novel0.0"] == True].copy()
        # All other stable materials (original False or unmappable) go to sun_false_stable
        novel_false_stable_df = stable_df[stable_df["is_novel0.0"] == False].copy()

        print(
            f"Total stable: {len(stable_df)}, Novel True Stable: {len(novel_true_stable_df)}, Novel False Stable: {len(novel_false_stable_df)}"
        )
        if len(stable_df) != (len(novel_true_stable_df) + len(novel_false_stable_df)):
            # This warning means some 'sun_0.0_bool' values were neither True nor False after mapping, which shouldn't happen with the current map_to_bool.
            # More likely, it would indicate an issue if stable_all_df itself had NaNs that were not handled by 'sun_0.0_bool' creation.
            # However, our map_to_bool defaults to False, so all rows in stable_all_df will have a sun_0.0_bool.
            print(
                "Info: Verification check: len(stable_all_df) == len(sun_true_stable_df) + len(sun_false_stable_df). If not, review 'sun_0.0' parsing."
            )
    else:
        raise ValueError("is_novel0.0 column not found in stable materials")

    # Precompute processed strings
    stable_df["processed_str"] = stable_df.progress_apply(
        process_crystal_string, axis=1
    )
    metastable_df["processed_str"] = metastable_df.progress_apply(
        process_crystal_string, axis=1
    )
    unstable_df["processed_str"] = unstable_df.progress_apply(
        process_crystal_string, axis=1
    )

    novel_true_stable_df["processed_str"] = novel_true_stable_df.progress_apply(
        process_crystal_string, axis=1
    )
    novel_false_stable_df["processed_str"] = novel_false_stable_df.progress_apply(
        process_crystal_string, axis=1
    )

    # 1. Create (Novel True Stable, Novel False Stable) pairs
    for _, novel_true_stable_row in novel_true_stable_df.iterrows():
        if len(novel_false_stable_df) == 0:
            break
        novel_false_stable_row = novel_false_stable_df.sample(1).iloc[0]
        temp = {
            "prompt": prompt,
            "chosen": novel_true_stable_row["processed_str"],
            "rejected": novel_false_stable_row["processed_str"],
        }
        preference_data.append(temp)

    # 1. Create (Stable, Metastable) pairs
    for _, stable_row in stable_df.iterrows():
        if len(metastable_df) == 0:
            break
        # Randomly sample one metastable per stable
        metastable_row = metastable_df.sample(1).iloc[0]
        temp = {
            "prompt": prompt,
            "chosen": stable_row["processed_str"],
            "rejected": metastable_row["processed_str"],
        }
        preference_data.append(temp)
        for _ in range(2):
            if len(unstable_df) == 0:
                break
            unstable_row = unstable_df.sample(1).iloc[0]
            temp = {
                "prompt": prompt,
                "chosen": stable_row["processed_str"],
                "rejected": unstable_row["processed_str"],
            }
            preference_data.append(temp)

    # 2. For every stable or metastable, create two (stable/metastable, unstable) pairs
    for _, row in metastable_df.iterrows():
        for _ in range(args.ratio):
            if len(unstable_df) == 0:
                break
            unstable_row = unstable_df.sample(1).iloc[0]
            temp = {
                "prompt": prompt,
                "chosen": row["processed_str"],
                "rejected": unstable_row["processed_str"],
            }
            preference_data.append(temp)

    return preference_data


def create_tiered_preference_dataset_sun(
    input_path, output_path, threshold=0.08, cif_column="relaxed_cif", args=None
):
    df = pd.read_csv(input_path)
    preference_data = []

    # Define prompt
    prompt = "Below is a description of a bulk material. "
    if args and args.conditions == "e_above_hull":
        prompt += "The energy above the convex hull is 0. "
    prompt += (
        "Generate a description of the lengths and angles of the lattice vectors "
        "and then the element type and coordinates for each atom within the lattice:\n"
    )
    tqdm.pandas()

    # Helper to get the correct string
    def process_crystal_string(row):
        if args and args.raw:
            return "\n".join(row["gen_str"].split("\n")[1:])
        else:
            return (
                get_crystal_string_wyckoff_pyx(row[cif_column])
                if args and args.wyckoff
                else get_crystal_string(row[cif_column])
            )

    df["processed_str"] = df.progress_apply(process_crystal_string, axis=1)

    # Classify by e_above_hull
    stable_all_df = df[df["e_above_hull"] <= 0].copy()
    metastable_df = df[
        (df["e_above_hull"] > 0) & (df["e_above_hull"] <= threshold)
    ].copy()
    unstable_df = df[df["e_above_hull"] > threshold].copy()

    sun_true_stable_df = pd.DataFrame(columns=stable_all_df.columns)
    sun_false_stable_df = pd.DataFrame(columns=stable_all_df.columns)

    if "sun_0.0" in stable_all_df.columns:
        sun_true_stable_df = stable_all_df[stable_all_df["sun_0.0"] == True].copy()
        # All other stable materials (original False or unmappable) go to sun_false_stable
        sun_false_stable_df = stable_all_df[stable_all_df["sun_0.0"] == False].copy()

        print(
            f"Total stable: {len(stable_all_df)}, Sun True Stable: {len(sun_true_stable_df)}, Sun False Stable: {len(sun_false_stable_df)}"
        )
        if len(stable_all_df) != (len(sun_true_stable_df) + len(sun_false_stable_df)):
            # This warning means some 'sun_0.0_bool' values were neither True nor False after mapping, which shouldn't happen with the current map_to_bool.
            # More likely, it would indicate an issue if stable_all_df itself had NaNs that were not handled by 'sun_0.0_bool' creation.
            # However, our map_to_bool defaults to False, so all rows in stable_all_df will have a sun_0.0_bool.
            print(
                "Info: Verification check: len(stable_all_df) == len(sun_true_stable_df) + len(sun_false_stable_df). If not, review 'sun_0.0' parsing."
            )
    else:
        print(
            "Warning: 'sun_0.0' column not found in stable materials. "
            "Sun-specific tiers (e.g., sun_true vs sun_false) will not be created. "
            "All stable materials will be treated as 'sun_false_stable' for subsequent tiering."
        )
        sun_false_stable_df = stable_all_df.copy()  # sun_true_stable_df remains empty

    print(f"Metastable: {len(metastable_df)}, Unstable: {len(unstable_df)}")

    # Tier 1: S_true vs S_false
    for _, s_true_row in sun_true_stable_df.iterrows():
        for _ in range(args.ratio):
            s_false_row = sun_false_stable_df.sample(1).iloc[0]
            preference_data.append(
                {
                    "prompt": prompt,
                    "chosen": s_true_row["processed_str"],
                    "rejected": s_false_row["processed_str"],
                }
            )

            m_row = metastable_df.sample(1).iloc[0]
            preference_data.append(
                {
                    "prompt": prompt,
                    "chosen": s_true_row["processed_str"],
                    "rejected": m_row["processed_str"],
                }
            )

            u_row = unstable_df.sample(1).iloc[0]
            preference_data.append(
                {
                    "prompt": prompt,
                    "chosen": s_true_row["processed_str"],
                    "rejected": u_row["processed_str"],
                }
            )

    # Tier 4: S_false vs M
    for _, s_false_row in sun_false_stable_df.iterrows():
        for _ in range(args.ratio):
            if metastable_df.empty:
                break
            m_row = metastable_df.sample(1).iloc[0]
            preference_data.append(
                {
                    "prompt": prompt,
                    "chosen": s_false_row["processed_str"],
                    "rejected": m_row["processed_str"],
                }
            )

    # Tier 5: S_false vs U
    if not sun_false_stable_df.empty and not unstable_df.empty:
        for _, s_false_row in sun_false_stable_df.iterrows():
            for _ in range(args.ratio):
                if unstable_df.empty:
                    break
                u_row = unstable_df.sample(1).iloc[0]
                preference_data.append(
                    {
                        "prompt": prompt,
                        "chosen": s_false_row["processed_str"],
                        "rejected": u_row["processed_str"],
                    }
                )

    # Tier 6: M vs U
    if not metastable_df.empty and not unstable_df.empty:
        for _, m_row in metastable_df.iterrows():
            for _ in range(args.ratio):
                if unstable_df.empty:
                    break
                u_row = unstable_df.sample(1).iloc[0]
                preference_data.append(
                    {
                        "prompt": prompt,
                        "chosen": m_row["processed_str"],
                        "rejected": u_row["processed_str"],
                    }
                )

    return preference_data


if __name__ == "__main__":
    parser = ArgumentParser(description="Create a preference dataset for fine-tuning.")
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input CSV file."
    )
    parser.add_argument(
        "--input_path_2",
        type=str,
        required=False,
        help="Path to additional input CSV file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output dataset.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.08, help="Threshold for e_above_hull."
    )
    parser.add_argument(
        "--cif_column",
        type=str,
        default="relaxed_cif",
        help="Column name for CIF data.",
    )
    parser.add_argument("--conditions", type=str, default=None)
    parser.add_argument("--raw", action="store_true", default=False)
    parser.add_argument("--wyckoff", action="store_true", default=False)
    parser.add_argument("--ratio", type=int, default=2)

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "basic",
            "tiered",
            "tiered2",
            "tieredSun",
            "tieredNovel",
            "sg",
            "sg_novel",
        ],
        default="basic",
        help="Dataset creation mode: 'basic' (uses create_preference_dataset_2), 'tiered' (uses create_tiered_preference_dataset), 'tiered2' (uses create_tiered_preference_dataset_v2), or 'tieredSun' (uses create_tiered_preference_dataset_sun). Default is 'basic'.",
    )
    parser.add_argument(
        "--mode_2",
        type=str,
        choices=["sg", "sg_novel"],
        default=None,
    )

    parser.add_argument(
        "--sg_14",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if args.mode == "tiered":
        preference_data = create_tiered_preference_dataset(
            input_path=args.input_path,
            threshold=args.threshold,
            cif_column=args.cif_column,
            args=args,
        )
    elif args.mode == "tiered2":
        preference_data = create_tiered_preference_dataset_v2(
            input_path=args.input_path,
            output_path=args.output_path,
            threshold=args.threshold,
            cif_column=args.cif_column,
            args=args,
        )
    elif args.mode == "tieredSun":
        preference_data = create_tiered_preference_dataset_sun(
            input_path=args.input_path,
            output_path=args.output_path,
            threshold=args.threshold,
            cif_column=args.cif_column,
            args=args,
        )
    elif args.mode == "tieredNovel":
        preference_data = create_tiered_preference_dataset_v2_novel(
            input_path=args.input_path,
            output_path=args.output_path,
            threshold=args.threshold,
            cif_column=args.cif_column,
            args=args,
        )
    elif args.mode == "sg":
        preference_data = create_sg_preference_dataset(
            input_path=args.input_path,
            threshold=args.threshold,
            cif_column=args.cif_column,
            args=args,
        )
    elif args.mode == "sg_novel":
        preference_data = create_sg_preference_dataset_novel(
            input_path=args.input_path,
            threshold=args.threshold,
            cif_column=args.cif_column,
            args=args,
        )
    else:
        preference_data = create_preference_dataset_2(
            input_path=args.input_path,
            threshold=args.threshold,
            cif_column=args.cif_column,
            args=args,
        )
    if args.mode_2:
        if args.mode_2 == "sg":
            print(f"Creating SG preference dataset from {args.input_path_2}")
            preference_data.extend(
                create_sg_preference_dataset(
                    input_path=args.input_path_2,
                    threshold=args.threshold,
                    cif_column=args.cif_column,
                    args=args,
                )
            )
        elif args.mode_2 == "sg_novel":
            print(f"Creating SG preference dataset from {args.input_path_2}")
            preference_data.extend(
                create_sg_preference_dataset_novel(
                    input_path=args.input_path_2,
                    threshold=args.threshold,
                    cif_column=args.cif_column,
                    args=args,
                )
            )

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(preference_data)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    dataset.save_to_disk(args.output_path)
    print(f"Preference dataset saved to {args.output_path}")


# choose sg 1, 15, 38, 119, 143, 194, 216
