import pandas as pd
import os

# Space group numbers mapped to symbols: 1-P1, 2-P-1, 6-Pm, 15-C2/c, 38-Amm2,
# 71-Immm, 119-I4/m2, 131-P4_2/mmc, 143-P3, 159-P31c, 187-P6m2, 194-P6_3/mmc, 204-Im3, 216-F4̅3m

def space_group_gen(space_group_nums, num_samples_per_sg=1000):
    print(len(space_group_nums))

    # Number of samples per space group
    num_samples_per_sg = 1000

    # Create a DataFrame with repeated space group numbers
    df = pd.DataFrame({"spacegroup_number": sum([[sg] * num_samples_per_sg for sg in space_group_nums], [])})
    print(df)

    # check if folder input_csv exists
    if not os.path.exists("cond_gen/input_csv"):
        os.mkdir("cond_gen/input_csv")
    # Save to CSV
    df.to_csv("cond_gen/input_csv/sg_in_short.csv", index=False)

    print("CSV file 'space_group_samples.csv' generated successfully.")


space_group_nums = [1, 2, 6, 15, 38, 71, 119, 131, 143, 159, 187, 194, 204, 216]
sg_nums_short = [1, 15, 38, 119, 143, 194, 216]
space_group_gen(sg_nums_short)