import os
import re
import warnings
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ============================================================
# TRAINING SCRIPT
# Uses BOTH ingredients.csv and datasets2.csv
# Keep both CSV files in the same folder as this script.
# ============================================================

DATASET_FILES = ["ingredients.csv", "datasets2.csv"]
MODEL_DIR = "models"
RANDOM_STATE = 42

def read_csv_safely(path):
    """Read a CSV while trying common encodings."""
    encodings = ["utf-8-sig", "cp1252", "latin1"]
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error

    if last_error:
        raise last_error
    return pd.read_csv(path)


def normalize_column_names(data):
    data = data.copy()
    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )
    return data


def normalize_category(value):
    """Convert common category spellings into consistent labels."""
    if pd.isna(value):
        return None

    value = str(value).strip().lower()
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    animal_values = {
        "animal", "animal based", "animal-derived", "animal derived",
        "non vegan", "nonvegan", "non vegan ingredient", "animal origin",
        "animal sourced", "animal source"
    }
    plant_values = {
        "plant", "plant based", "plant-derived", "plant derived",
        "vegan", "plant origin", "plant sourced", "plant source"
    }
    uncertain_values = {
        "uncertain", "unknown", "not sure", "ambiguous", "unverified"
    }

    if value in animal_values:
        return "Animal-Based"
    if value in plant_values:
        return "Plant-Based"
    if value in uncertain_values:
        return "Uncertain"

    # Handle some common longer labels.
    if "animal" in value and any(word in value for word in ("based", "derived", "sourced", "origin")):
        return "Animal-Based"
    if "plant" in value and any(word in value for word in ("based", "derived", "sourced", "origin")):
        return "Plant-Based"
    if "non vegan" in value or "nonvegan" in value:
        return "Animal-Based"
    if value == "true":
        return "Plant-Based"
    if value == "false":
        return "Animal-Based"

    return None


def find_ingredient_column(data):
    """Find the most likely ingredient-name column."""
    preferred = [
        "ingredient", "ingredients", "actual_ingredient",
        "ingredient_name", "ingredient_text", "name"
    ]
    for column in preferred:
        if column in data.columns:
            return column

    matches = [column for column in data.columns if "ingredient" in column]
    if matches:
        # Prefer the shortest matching name, often the main ingredient column.
        return sorted(matches, key=len)[0]
    return None


def derive_category_from_columns(data):
    """
    Use a category column when present. Otherwise, try common vegan/non-vegan
    indicator columns from older datasets.
    """
    if "category" in data.columns:
        return data["category"].apply(normalize_category)

    # If a column explicitly says animal/plant source, try it.
    for column in ("source_type", "source_category", "origin", "source"):
        if column in data.columns:
            labels = data[column].apply(normalize_category)
            if labels.notna().sum() > 0:
                return labels

    # Handle datasets that have vegan/non-vegan columns.
    vegan_columns = [c for c in data.columns if c in ("vegan", "is_vegan", "vegan_status")]
    nonvegan_columns = [c for c in data.columns if c in ("non_vegan", "nonvegan", "is_non_vegan")]

    def truthy(value):
        if pd.isna(value):
            return False
        return str(value).strip().lower() in {
            "1", "true", "yes", "y", "vegan", "non-vegan", "non vegan"
        }

    labels = pd.Series([None] * len(data), index=data.index, dtype="object")

    if vegan_columns:
        col = vegan_columns[0]
        for idx, value in data[col].items():
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "y", "vegan"}:
                labels.loc[idx] = "Plant-Based"
            elif text in {"0", "false", "no", "n", "non-vegan", "non vegan"}:
                labels.loc[idx] = "Animal-Based"

    if nonvegan_columns:
        col = nonvegan_columns[0]
        for idx, value in data[col].items():
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "y"}:
                labels.loc[idx] = "Animal-Based"
            elif text in {"0", "false", "no", "n"} and pd.isna(labels.loc[idx]):
                labels.loc[idx] = "Plant-Based"

    return labels


def prepare_dataset(path):
    print(f"\nReading: {path}")
    data = read_csv_safely(path)
    data = normalize_column_names(data)
    print("Columns found:", data.columns.tolist())

    ingredient_column = find_ingredient_column(data)
    if ingredient_column is None:
        raise ValueError(
            f"Could not identify an ingredient-name column in {path}. "
            f"Columns found: {data.columns.tolist()}"
        )

    categories = derive_category_from_columns(data)
    if categories is None or categories.notna().sum() == 0:
        raise ValueError(
            f"Could not identify usable category labels in {path}. "
            "The file needs a category column (Animal-Based/Plant-Based/Uncertain) "
            "or recognizable vegan/non-vegan labels."
        )

    result = pd.DataFrame({
        "ingredient": data[ingredient_column],
        "category": categories
    })

    result["ingredient"] = (
        result["ingredient"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    result = result[
        result["ingredient"].ne("")
        & result["ingredient"].ne("nan")
        & result["category"].notna()
    ].copy()

    print(f"Usable labelled records from {path}: {len(result)}")
    print("Category counts:")
    print(result["category"].value_counts().to_string())
    return result


def main():
    print("=" * 65)
    print("ANIMAL-DERIVED INGREDIENT CLASSIFIER TRAINING")
    print("This script combines ingredients.csv AND datasets2.csv")
    print("=" * 65)

    missing_files = [file for file in DATASET_FILES if not os.path.exists(file)]
    if missing_files:
        raise FileNotFoundError(
            "These dataset file(s) were not found in the current folder: "
            + ", ".join(missing_files)
            + "\nPut both CSV files in the same folder as train_model.py."
        )

    datasets = [prepare_dataset(path) for path in DATASET_FILES]
    df = pd.concat(datasets, ignore_index=True)

    print("\nCombined labelled records before cleaning:", len(df))

    # Normalize category labels and remove duplicate ingredient/category pairs.
    df = df.drop_duplicates(subset=["ingredient", "category"]).copy()

    # If the same normalized ingredient has conflicting labels across the files,
    # exclude it rather than training on contradictory labels.
    label_counts = df.groupby("ingredient")["category"].nunique()
    conflicting = label_counts[label_counts > 1].index
    if len(conflicting):
        print(
            f"\nRemoving {len(conflicting)} ingredient names with conflicting "
            "labels across the datasets."
        )
        df = df[~df["ingredient"].isin(conflicting)].copy()

    df = df.drop_duplicates(subset=["ingredient"]).reset_index(drop=True)

    print("\nFinal records available for training:", len(df))
    print("\nFinal category distribution:")
    print(df["category"].value_counts().to_string())

    if df["category"].nunique() < 2:
        raise ValueError(
            "At least two categories are needed. Check the category labels "
            "in both CSV files."
        )

    class_counts = df["category"].value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "Each category needs at least two examples for train/test splitting."
        )

    X = df["ingredient"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    models = {
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                sublinear_tf=True,
                max_features=30000,
                min_df=1
            )),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
                max_features="sqrt"
            ))
        ]),
        "SVM": Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                sublinear_tf=True,
                max_features=30000,
                min_df=1
            )),
            ("classifier", LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ])
    }

    results = []

    for model_name, model in models.items():
        print("\n" + "=" * 65)
        print("Training:", model_name)
        print("Training rows:", len(X_train), "| Test rows:", len(X_test))

        with warnings.catch_warnings():
            warnings.simplefilter("default")
            model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        print(f"\nAccuracy:  {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}% (weighted)")
        print(f"Recall:    {recall * 100:.2f}% (weighted)")
        print(f"F1-score:  {f1 * 100:.2f}% (weighted)")

        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            labels=sorted(y.unique()),
            zero_division=0
        ))

        print("Confusion matrix (labels:", sorted(y.unique()), "):")
        print(confusion_matrix(y_test, predictions, labels=sorted(y.unique())))

        filename = (
            "models/random_forest.pkl"
            if model_name == "Random Forest"
            else "models/svm.pkl"
        )
        joblib.dump(model, filename)
        print("Saved model to:", filename)

        results.append({
            "Model": model_name,
            "Accuracy_Percent": round(accuracy * 100, 2),
            "Weighted_Precision_Percent": round(precision * 100, 2),
            "Weighted_Recall_Percent": round(recall * 100, 2),
            "Weighted_F1_Percent": round(f1 * 100, 2),
            "Training_Rows": len(X_train),
            "Test_Rows": len(X_test)
        })

    comparison = pd.DataFrame(results)
    comparison.to_csv("model_comparison.csv", index=False)

    # Save a compact metadata file for reference.
    metadata = {
        "dataset_files": DATASET_FILES,
        "label_classes": sorted(df["category"].unique().tolist()),
        "training_records_after_cleaning": int(len(df)),
        "random_state": RANDOM_STATE
    }
    joblib.dump(metadata, "models/training_metadata.pkl")

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print(comparison.to_string(index=False))
    print("\nTraining completed successfully.")
    print("Saved models in the 'models' folder.")
    print("Saved comparison to model_comparison.csv")
    print(
        "\nImportant: test scores can be optimistic if the dataset contains "
        "very similar ingredient-name variants. Check model behavior on real "
        "product-label ingredients before relying on predictions."
    )


if __name__ == "__main__":
    main()
