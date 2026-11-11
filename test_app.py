import pytest
import pandas as pd

import app


# ==========================================
# TEST 1: TEXT NORMALIZATION
# ==========================================

def test_normalize_text():

    result = app.normalize_text("Aloe-Vera!")

    assert result == "aloe vera"


# ==========================================
# TEST 2: INGREDIENT SPLITTING
# ==========================================

def test_split_ingredients():

    text = "Water, Glycerin, Aloe Vera, Collagen"

    result = app.split_ingredients(text)

    assert len(result) == 4

    assert "Water" in result
    assert "Glycerin" in result
    assert "Aloe Vera" in result
    assert "Collagen" in result


# ==========================================
# TEST 3: ANIMAL-BASED CLASSIFICATION
# ==========================================

def test_animal_classification():

    result = app.classify_ingredient(
        "Gelatin",
        pd.DataFrame()
    )

    assert result["Category"] == "Animal-Based"


# ==========================================
# TEST 4: PLANT-BASED CLASSIFICATION
# ==========================================

def test_plant_classification():

    result = app.classify_ingredient(
        "Aloe Vera",
        pd.DataFrame()
    )

    assert result["Category"] == "Plant-Based"


# ==========================================
# TEST 5: UNCERTAIN CLASSIFICATION
# ==========================================

def test_uncertain_classification(monkeypatch):

    monkeypatch.setattr(
        app,
        "load_ml_models",
        lambda: (None, None, None)
    )

    result = app.classify_ingredient(
        "Unknown Ingredient XYZ",
        pd.DataFrame()
    )

    assert result["Category"] == "Uncertain"


# ==========================================
# TEST 6: DATASET CLASSIFICATION
# ==========================================

def test_dataset_classification():

    dataset = pd.DataFrame({

        "ingredient": ["Glycerin"],

        "category": ["Plant-Based"],

        "explanation": [
            "Obtained from vegetable sources."
        ]

    })

    result = app.classify_ingredient(
        "Glycerin",
        dataset
    )

    assert result["Category"] == "Plant-Based"


# ==========================================
# TEST 7: PDF REPORT GENERATION
# ==========================================

def test_pdf_generation():

    results = [

        {
            "Ingredient": "Gelatin",

            "Category": "Animal-Based",

            "Explanation": "Obtained from animal collagen."
        },

        {
            "Ingredient": "Aloe Vera",

            "Category": "Plant-Based",

            "Explanation": "Obtained from plants."
        }

    ]

    pdf = app.generate_pdf_report(results)

    assert pdf.startswith(b"%PDF")

    assert len(pdf) > 0


# ==========================================
# TEST 8: EMPTY PDF REPORT
# ==========================================

def test_empty_pdf_report():

    with pytest.raises(ValueError):

        app.generate_pdf_report([])


# ==========================================
# TEST 9: DUPLICATE INGREDIENT REMOVAL
# ==========================================

def test_duplicate_ingredients():

    text = "Water, Glycerin, Water, Aloe Vera"

    result = app.split_ingredients(text)

    assert len(result) == 3


# ==========================================
# TEST 10: CATEGORY IDENTIFICATION
# ==========================================

def test_category_identification():

    assert (
        app.get_category_name("Animal-Based")
        == "Animal-Based"
    )

    assert (
        app.get_category_name("Plant-Based")
        == "Plant-Based"
    )

    assert (
        app.get_category_name("Unknown")
        == "Uncertain"
    )   