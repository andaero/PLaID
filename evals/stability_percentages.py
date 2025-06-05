import pandas as pd


def calculate_stability_percentages(file_path, threshold):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Ensure the e_above_hull column exists
    if "e_above_hull" not in df.columns:
        raise ValueError("The input CSV file must contain an 'e_above_hull' column.")

    # Total number of crystals
    total_crystals = len(df)

    # Count metastable crystals (e_above_hull < 0.1)
    metastable_count = df[df["e_above_hull"] <= 0.1].shape[0] / 100

    metastable_08_count = df[df["e_above_hull"] <= threshold].shape[0] / 100

    # Count stable crystals (e_above_hull < 0)
    stable_count = df[df["e_above_hull"] <= 0].shape[0] / 100

    average_e_above_hull = df["e_above_hull"].mean()

    # Calculate percentages
    metastable_percentage = (metastable_count / total_crystals) * 100
    stable_percentage = (stable_count / total_crystals) * 100
    metastable_08_percentage = (metastable_08_count / total_crystals) * 100
    return metastable_count, metastable_08_count, stable_count, average_e_above_hull


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="evals/results/alex_8b_wyckoff_210k_temp_0.7_eqv2_batched_ehull_results.csv",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.08,
    )
    args = parser.parse_args()

    file_path = args.file
    # file_path = "evals/results/alex_8b_wyckoff_210k_temp_0.7_ehull_results.csv"
    try:
        (
            metastable_percentage,
            metastable_08_percentage,
            stable_percentage,
            average_e_above_hull,
        ) = calculate_stability_percentages(file_path, args.threshold)
        print(
            f"Percentage of metastable crystals (e_above_hull < 0.1): {metastable_percentage:.2f}%"
        )
        print(
            f"Percentage of metastable crystals (e_above_hull < {args.threshold}): {metastable_08_percentage:.2f}%"
        )
        print(
            f"Percentage of stable crystals (e_above_hull < 0): {stable_percentage:.2f}%"
        )
        print(f"Average e_above_hull: {average_e_above_hull:.2f}")
    except Exception as e:
        print(f"An error occurred: {e}")
