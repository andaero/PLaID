import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pyxtal import pyxtal
from pyxtal.symmetry import Group
from pyxtal.lattice import Lattice


def filter_correct_spacegroup(df, tol):
    def analyze(x):
        structure = Structure.from_str(x["cif"], fmt="cif")
        try:
            sga = SpacegroupAnalyzer(structure, tol)
            sg = sga.get_space_group_number()
        except:
            return False
        import re

        sg_des = re.search(r"The spacegroup number is (\d+)\.", x["gen_str"]).group(1)
        return sg == int(sg_des)

    df["is_correct"] = df.apply(analyze, axis=1)

    correct_count = df["is_correct"].sum()
    incorrect_count = len(df) - correct_count
    total_count = len(df)

    accuracy = correct_count / total_count
    inaccuracy = incorrect_count / total_count

    summary = {
        "total": total_count,
        "correct": correct_count,
        "incorrect": incorrect_count,
        "accuracy": accuracy,
        "inaccuracy": inaccuracy,
    }
    print(summary)

    filtered_df = df[df["is_correct"]]

    filtered_df = filtered_df.drop(columns=["is_correct"])

    return filtered_df


def parse_sg_col(df):
    # pass in dataframe with gen_str col, then parse out from the text "The spacegroup number is 1." where 1 is the number
    df["spacegroup_number"] = df["gen_str"].str.extract(
        r"The spacegroup number is (\d+)\."
    )
    # print num of samples per space group
    print(df["spacegroup_number"].value_counts())

    return df


def seperate_dfs_by_condition(folder_name, condition):
    file_name = f"{folder_name}.csv"
    df = pd.read_csv(file_name)
    # create folder for the file and then save multiple dfs to it by condition
    if not os.path.exists(folder_name):
        print(f"Creating folder {folder_name}")
        os.makedirs(folder_name)
    for condition_value, group_df in df.groupby(condition):
        print(f"Saving {condition_value} to {folder_name}/{condition_value}.csv")
        group_df.to_csv(f"{folder_name}/{condition_value}.csv", index=False)

def get_actual_sg(x, strict=False):
    structure = Structure.from_str(x["relaxed_cif"], fmt="cif")

    pyx = pyxtal()
    try:
        if strict:
            pyx.from_seed(structure, tol=0.01)
        else:
            pyx.from_seed(structure, tol=0.1)
    except:
        print("failed pyxtal 0.01 trying 0.0001 tol")
        pyx.from_seed(structure, tol=0.0001)
    return pyx.group.number

def apply_parallel(func, df, num_workers, strict=False):
    results = []
    # Ensure tqdm is available or provide a fallback
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs):
            return iterable

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use a list comprehension to pass rows to the executor
        futures = [executor.submit(func, row, strict) for _, row in df.iterrows()]
        for future in tqdm(futures, total=len(df), desc="Processing", ncols=100):
            result = future.result()  # Correctly get the result from the future
            if result is not None:
                results.append(result)
    print(f"Processed {len(results)} rows successfully.")
    return results

def get_real_space_group(df, num_workers=None, strict=False):
    """
    Calculates the actual space group for each structure in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing structure data. 
                           It must have a 'relaxed_cif' column.
        num_workers (int, optional): Number of worker processes for parallel execution. 
                                     Defaults to None (uses os.cpu_count()).
        strict (bool, optional): Whether to use strict tolerance (0.01) for pyxtal. 
                                 Defaults to False (uses 0.1).

    Returns:
        pd.DataFrame: The input DataFrame with an added 'actual_spacegroup' column.
    """
    if 'relaxed_cif' not in df.columns:
        raise ValueError("DataFrame must contain a 'relaxed_cif' column.")
    
    # check if actual_spacegroup is already in df
    if 'actual_spacegroup' in df.columns:
        print("actual_spacegroup already in df, skipping")
        return df
    
    # Pass get_actual_sg directly, along with the strict parameter
    if strict:
        print('Using strict tol 0.01')
    else:
        print('Using tol 0.1')
    df["actual_spacegroup"] = apply_parallel(get_actual_sg, df, num_workers, strict=strict)
    return df

# df = pd.read_csv("new_llm_samples/llm8b_wyckoff_sg_samples_temp_0.7.csv")
# df = filter_correct_spacegroup(df, 0.1)
# df = parse_sg_col(df)

# df.to_csv("new_llm_samples/llm8b_wyckoff_sg_samples_temp_0.7.csv", index=False)

# seperate_dfs_by_condition(
#     "new_llm_samples/qwen_7b_wyckoff_sg_temp_0.7", "spacegroup_number"
# )

# df = pd.read_csv("evals/results/qwen_7b_wyckoff_temp_0.7_eqv2_batched_ehull_results.csv")
# df = get_real_space_group(df)
# print(df.head())

# for file in os.listdir("evals/results/qwen_7b_wyckoff_sg/"):
#     print(f"Processing {file}")
#     df = pd.read_csv(f"evals/results/qwen_7b_wyckoff_sg/{file}")
#     df = get_real_space_group(df, strict=True)
#     df.to_csv(f"evals/results/qwen_7b_wyckoff_sg/{file}", index=False)