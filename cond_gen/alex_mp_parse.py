import pandas as pd
from pymatgen.symmetry.groups import SpaceGroup
from tqdm import tqdm

def space_group_number_for_symbol(symbol: str) -> int:
    return SpaceGroup(symbol).int_number

def change_col_names(df: pd.DataFrame, export=False, new_name="") -> pd.DataFrame:
    # change energy_above_hull to e_above_hull, and 
    new_col_names = {
        "energy_above_hull": "e_above_hull",
        "reduced_formula": "pretty_formula",
        "dft_band_gap": "band_gap",
    }
    df = df.rename(columns=new_col_names)

    if "spacegroup.number" not in df.columns:
        # calc space_group.number from symbol "space_group"
        tqdm.pandas(desc="Converting space groups")

        df["spacegroup.number"] = df["space_group"].progress_apply(space_group_number_for_symbol)
    if export:
        df.to_csv(f"alex_mp_20/{new_name}.csv", index=False)
    return df

df = pd.read_csv("alex_mp_20/train.csv")

df = change_col_names(df, export=True, new_name="train")
print(df.columns)
# print(df.head())

df_val = pd.read_csv("alex_mp_20/val.csv")
df_val = change_col_names(df_val, export=True, new_name="val")
print(df_val.columns)
# print(df_val.head())