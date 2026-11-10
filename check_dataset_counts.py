import pandas as pd
import os

# Dataset files
dataset_files = [
    "ingredients.csv",
    "datasets2.csv"
]

all_data = []

print("\n==========================================")
print("  ANIMAL INGREDIENT DATASET SUMMARY")
print("==========================================")

for file in dataset_files:

    if not os.path.exists(file):
        print(f"\nFile not found: {file}")
        continue

    try:
        df = pd.read_csv(file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file, encoding="latin1")

    print(f"\nDataset: {file}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Find the category column
    category_column = next(
        (
            col for col in df.columns
            if col.strip().lower() in [
                "category",
                "classification",
                "ingredient_type",
                "type",
                "label"
            ]
        ),
        None
    )

    if category_column:
        print("\nCategory counts in this dataset:")
        print(df[category_column].value_counts(dropna=False).to_string())

        df["standard_category"] = (
            df[category_column]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("_", " ", regex=False)
            .str.replace("-", " ", regex=False)
        )

        def standardize_category(value):
            if value in [
                "animal based", "animal", "non vegan",
                "non vegan ingredient", "animal derived"
            ]:
                return "Animal-Based"

            if value in [
                "plant based", "plant", "vegan",
                "plant derived"
            ]:
                return "Plant-Based"

            if value in [
                "uncertain", "unknown", "unclear",
                "undetermined"
            ]:
                return "Uncertain"

            return "Other / Unrecognized"

        df["standard_category"] = (
            df["standard_category"].apply(standardize_category)
        )

        all_data.append(df[["standard_category"]])

    else:
        print(
            "\nNo recognized category column was found. "
            "Check the column names printed above."
        )

# Combine the available category records
if all_data:

    combined = pd.concat(all_data, ignore_index=True)

    print("\n==========================================")
    print("       COMBINED DATASET TOTALS")
    print("==========================================")

    counts = combined["standard_category"].value_counts()

    for category in [
        "Animal-Based",
        "Plant-Based",
        "Uncertain",
        "Other / Unrecognized"
    ]:
        print(f"{category}: {counts.get(category, 0):,}")

    print("------------------------------------------")
    print(f"TOTAL RECORDS: {len(combined):,}")
    print("==========================================")

    print(
        "\nNote: These are row counts across both files. "
        "Duplicate ingredients may be counted more than once."
    )

else:
    print("\nNo category data could be counted.")
    print("Please check the dataset filenames and columns.")