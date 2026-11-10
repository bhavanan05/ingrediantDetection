import streamlit as st
import pandas as pd
import joblib
import easyocr
import cv2
import numpy as np
import re
import os
from PIL import Image
from io import BytesIO
from difflib import SequenceMatcher

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Ingredient Origin Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PROFESSIONAL CSS DESIGN
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
        color: #111827;
    }

    html, body, [class*="css"] {
        color: #111827;
        font-family: Arial, sans-serif;
    }

    h1, h2, h3, h4, p, label {
        color: #111827 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #F4F7F5;
    }

    [data-testid="stSidebar"] * {
        color: #111827 !important;
    }

    .hero {
        background: linear-gradient(120deg, #F0F7F1, #FFFFFF);
        border: 1px solid #DCE9DE;
        border-radius: 18px;
        padding: 30px;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: #173D2B !important;
        font-size: 36px;
        margin-bottom: 10px;
    }

    .hero p {
        font-size: 16px;
        color: #475569 !important;
    }

    .section-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
    }

    .ingredient-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2E7D50;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }

    .animal {
        color: #B91C1C !important;
        font-weight: bold;
    }

    .plant {
        color: #15803D !important;
        font-weight: bold;
    }

    .uncertain {
        color: #B45309 !important;
        font-weight: bold;
    }

    div.stButton > button {
        background-color: #216E45;
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }

    div.stButton > button:hover {
        background-color: #174D30;
        color: white !important;
    }

    .footer {
        text-align: center;
        color: #64748B;
        padding: 20px;
        border-top: 1px solid #E2E8F0;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD PRIVATE INGREDIENT REFERENCE DATA
# --------------------------------------------------

@st.cache_data
def load_ingredient_data():

    possible_files = [
        "ingredients.csv",
        "datasetsssssfinal.csv"
    ]

    for filename in possible_files:
        try:
            df = pd.read_csv(
                filename,
                encoding="cp1252",
                on_bad_lines="skip"
            )

            if len(df.columns) < 2:
                df = pd.read_csv(
                    filename,
                    encoding="utf-8",
                    on_bad_lines="skip"
                )

            df.columns = [
                str(col).strip().lower()
                for col in df.columns
            ]

            ingredient_col = next(
                (
                    col for col in df.columns
                    if "ingredient" in col
                ),
                None
            )

            category_col = next(
                (
                    col for col in df.columns
                    if "category" in col
                    or "classification" in col
                ),
                None
            )

            explanation_col = next(
                (
                    col for col in df.columns
                    if "explanation" in col
                    or "reason" in col
                    or "source" in col
                ),
                None
            )

            # Support files with unexpected headers.
            if ingredient_col is None and len(df.columns) >= 1:
                ingredient_col = df.columns[0]

            if category_col is None and len(df.columns) >= 2:
                category_col = df.columns[1]

            if ingredient_col and category_col:

                selected = pd.DataFrame()

                selected["ingredient"] = (
                    df[ingredient_col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

                selected["category"] = (
                    df[category_col]
                    .fillna("Uncertain")
                    .astype(str)
                    .str.strip()
                )

                if explanation_col:
                    selected["explanation"] = (
                        df[explanation_col]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                    )
                else:
                    selected["explanation"] = ""

                selected = selected[
                    selected["ingredient"] != ""
                ]

                selected = selected.drop_duplicates(
                    subset=["ingredient"],
                    keep="first"
                )

                if not selected.empty:
                    return selected

        except FileNotFoundError:
            continue

        except Exception:
            continue

    return pd.DataFrame(
        columns=["ingredient", "category", "explanation"]
    )


# --------------------------------------------------
# NORMALIZE INGREDIENT TEXT
# --------------------------------------------------

def normalize_text(text):

    text = str(text).lower().strip()

    text = re.sub(r"[^a-z0-9\s-]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text


# --------------------------------------------------
# INGREDIENT CLASSIFICATION
# --------------------------------------------------

def classify_ingredient(ingredient, reference_data):

    ingredient = ingredient.strip()

    normalized_ingredient = normalize_text(ingredient)

    if not normalized_ingredient:
        return {
            "Ingredient": ingredient,
            "Classification": "Uncertain",
            "Explanation": "No ingredient text was provided."
        }

    # If the reference file is unavailable, use a small
    # built-in fallback dictionary.

    fallback = {
        "gelatin": (
            "Animal-Based",
            "Gelatin is generally derived from animal collagen."
        ),
        "collagen": (
            "Animal-Based",
            "Collagen commonly comes from animal connective tissue."
        ),
        "lanolin": (
            "Animal-Based",
            "Lanolin is obtained from sheep's wool."
        ),
        "beeswax": (
            "Animal-Based",
            "Beeswax is produced by honeybees."
        ),
        "carmine": (
            "Animal-Based",
            "Carmine is a red pigment derived from cochineal insects."
        ),
        "tallow": (
            "Animal-Based",
            "Tallow is rendered animal fat."
        ),
        "keratin": (
            "Animal-Based",
            "Cosmetic keratin is commonly sourced from animal materials."
        ),
        "honey": (
            "Animal-Based",
            "Honey is produced by bees."
        ),
        "milk": (
            "Animal-Based",
            "Milk is an animal-derived ingredient."
        ),
        "aloe vera": (
            "Plant-Based",
            "Aloe vera is a plant-derived ingredient."
        ),
        "coconut oil": (
            "Plant-Based",
            "Coconut oil is obtained from coconuts."
        ),
        "olive oil": (
            "Plant-Based",
            "Olive oil is obtained from olives."
        ),
        "shea butter": (
            "Plant-Based",
            "Shea butter is derived from shea tree nuts."
        ),
        "glycerin": (
            "Uncertain",
            "Glycerin may come from plant, animal, or synthetic sources."
        ),
        "glycerol": (
            "Uncertain",
            "Glycerol may come from plant, animal, or synthetic sources."
        ),
        "squalene": (
            "Uncertain",
            "Squalene can come from animal, plant, or other sources."
        ),
        "stearic acid": (
            "Uncertain",
            "Stearic acid may be derived from animal or plant fats."
        ),
        "cetyl alcohol": (
            "Uncertain",
            "The source can vary depending on the manufacturer."
        )
    }

    # Check the built-in dictionary first.

    for known_ingredient, result in fallback.items():

        if normalized_ingredient == known_ingredient:

            return {
                "Ingredient": ingredient,
                "Classification": result[0],
                "Explanation": result[1]
            }

    # Match against the private reference file.

    if not reference_data.empty:

        reference = reference_data.copy()

        reference["normalized"] = (
            reference["ingredient"]
            .apply(normalize_text)
        )

        # Exact match.

        exact = reference[
            reference["normalized"] == normalized_ingredient
        ]

        if not exact.empty:

            row = exact.iloc[0]

            category = str(row["category"]).strip()

            explanation = str(row["explanation"]).strip()

            return {
                "Ingredient": ingredient,
                "Classification": category,
                "Explanation": (
                    explanation
                    if explanation
                    else "Classification retrieved from the ingredient reference."
                )
            }

        # Match ingredient names when one is contained in the other.

        matches = reference[
            reference["normalized"].apply(
                lambda value:
                    len(value) >= 4
                    and (
                        value in normalized_ingredient
                        or normalized_ingredient in value
                    )
            )
        ]

        if len(matches) == 1:

            row = matches.iloc[0]

            return {
                "Ingredient": ingredient,
                "Classification": str(row["category"]),
                "Explanation": (
                    str(row["explanation"])
                    if str(row["explanation"]).strip()
                    else "A related ingredient was found in the reference."
                )
            }

    # Use the trained SVM model for ingredients not matched above.
    # The model file is optional; the app can still run without it.
    try:
        model = load_trained_model()
        if model is not None:
            prediction = str(model.predict([ingredient])[0]).strip()
            allowed_categories = {
                "Animal-Based", "Plant-Based", "Uncertain"
            }

            if prediction in allowed_categories:
                return {
                    "Ingredient": ingredient,
                    "Classification": prediction,
                    "Explanation": (
                        "Predicted by the trained SVM machine-learning model. "
                        "This is an estimate and may not reflect the supplier's actual source."
                    )
                }
    except Exception:
        pass

    return {
        "Ingredient": ingredient,
        "Classification": "Uncertain",
        "Explanation": (
            "The ingredient could not be confidently identified. "
            "Check the manufacturer's ingredient source."
        )
    }


@st.cache_resource
def load_trained_model():
    """Load the trained SVM pipeline if it is present."""
    model_path = os.path.join("models", "svm.pkl")
    if not os.path.isfile(model_path):
        return None
    return joblib.load(model_path)


# --------------------------------------------------
# EASY OCR
# --------------------------------------------------

@st.cache_resource
def load_ocr_reader():

    return easyocr.Reader(
        ["en"],
        gpu=False
    )


def extract_text(image):

    reader = load_ocr_reader()

    image_array = np.array(image.convert("RGB"))

    image_array = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    results = reader.readtext(
        image_array,
        detail=0,
        paragraph=True
    )

    return "\n".join(results)


# --------------------------------------------------
# SPLIT INGREDIENTS
# --------------------------------------------------

def split_ingredients(text):

    text = text.replace("\n", ",")

    parts = re.split(r"[,;]+", text)

    ingredients = []

    for part in parts:

        part = part.strip()

        part = re.sub(
            r"^(ingredients?|composition)\s*:\s*",
            "",
            part,
            flags=re.IGNORECASE
        )

        part = part.strip(" .:-")

        if len(part) >= 2:
            ingredients.append(part)

    # Remove duplicate names while preserving order.

    final_ingredients = []
    seen = set()

    for ingredient in ingredients:

        key = normalize_text(ingredient)

        if key not in seen:
            final_ingredients.append(ingredient)
            seen.add(key)

    return final_ingredients


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🌿 Ingredient Origin Detector</h1>
    <p>
        Discover the possible origin of ingredients in your
        personal care products using Optical Character Recognition
        and Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🌿 Ingredient Detector")

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Ingredient Scanner",
            "About"
        ]
    )

    st.markdown("---")

    st.markdown("### How it works")

    st.write("1. Upload a product label or take a photo.")

    st.write("2. Extract ingredient text using OCR.")

    st.write("3. Review and edit the extracted text.")

    st.write("4. Analyze the ingredients.")

    st.write("5. Download the report.")


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.subheader("Welcome to Ingredient Origin Detection")

    st.write(
        "Understand whether ingredients may be animal-derived "
        "or plant-derived before choosing a product."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>📷 Image Recognition</h3>
            <p>Upload a label or capture one using your camera.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-card">
            <h3>🔎 Ingredient Analysis</h3>
            <p>Extract text and identify possible ingredient origins.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="section-card">
            <h3>📄 Download Report</h3>
            <p>Save your ingredient analysis as a CSV file.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Ingredient classifications are estimates. "
        "The same ingredient may have different sources depending "
        "on the manufacturer."
    )


# --------------------------------------------------
# INGREDIENT SCANNER PAGE
# --------------------------------------------------

elif page == "Ingredient Scanner":

    st.subheader("Scan Your Product Label")
    st.write("Upload a product-label image or capture one using your camera.")

    upload_col, camera_col = st.columns(2)

    with upload_col:
        st.markdown("### 📁 Upload an Image")
        uploaded_file = st.file_uploader(
            "Choose an image from your device",
            type=["png", "jpg", "jpeg", "webp"],
            key="label_upload"
        )

    with camera_col:
        st.markdown("### 📸 Capture with Camera")
        camera_file = st.camera_input(
            "Take a photo of the ingredient label",
            key="label_camera"
        )

    image_file = None
    source_choice = None

    if uploaded_file is not None and camera_file is not None:
        source_choice = st.radio(
            "Choose which image to analyze",
            ["Uploaded image", "Camera image"],
            horizontal=True,
            key="image_source_choice"
        )
        image_file = uploaded_file if source_choice == "Uploaded image" else camera_file
    elif uploaded_file is not None:
        image_file = uploaded_file
    elif camera_file is not None:
        image_file = camera_file

    if image_file is not None:
        try:
            image = Image.open(image_file).convert("RGB")
            st.image(
                image,
                caption="Selected product label",
                use_container_width=True
            )

            if st.button("🔍 Extract Ingredient Text", key="extract_label_text"):
                with st.spinner("Reading the label. This may take a moment..."):
                    extracted_text = extract_text(image)

                st.session_state["extracted_text"] = extracted_text
                st.session_state["ingredient_text_editor"] = extracted_text
                st.session_state.pop("analysis_results", None)

                if extracted_text.strip():
                    st.success("Text extraction completed. Review the text below.")
                else:
                    st.warning("No readable text was found. Try a clearer, closer image with good lighting.")

        except Exception as error:
            st.error("The selected image could not be opened or processed. Please try another image.")
            st.caption(f"Technical detail: {error}")

    st.markdown("### ✏️ Review Extracted Text")
    st.caption("Check the OCR result and correct any mistakes before analyzing.")

    if "ingredient_text_editor" not in st.session_state:
        st.session_state["ingredient_text_editor"] = st.session_state.get("extracted_text", "")

    ingredient_text = st.text_area(
        "Extracted ingredient text",
        height=180,
        placeholder="Extracted text from your label will appear here...",
        key="ingredient_text_editor"
    )

    if st.button("🧪 Analyze Ingredients", key="analyze_ingredients"):
        ingredients = split_ingredients(ingredient_text)

        if not ingredients:
            st.warning("First upload or capture a label and extract its text.")
        else:
            reference_data = load_ingredient_data()
            results = []
            progress = st.progress(0)

            for index, ingredient in enumerate(ingredients):
                results.append(classify_ingredient(ingredient, reference_data))
                progress.progress(int(((index + 1) / len(ingredients)) * 100))

            progress.empty()
            st.session_state["analysis_results"] = results
            st.success("Ingredient analysis completed!")

    if "analysis_results" in st.session_state:
        results = st.session_state["analysis_results"]
        results_df = pd.DataFrame(results)

        st.markdown("---")
        st.subheader("📊 Analysis Results")

        animal_count = sum(1 for item in results if "animal" in item["Classification"].lower())
        plant_count = sum(1 for item in results if "plant" in item["Classification"].lower())
        uncertain_count = len(results) - animal_count - plant_count

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Animal-Based", animal_count)
        metric2.metric("Plant-Based", plant_count)
        metric3.metric("Uncertain", uncertain_count)

        for item in results:
            category = str(item.get("Classification", "Uncertain"))
            category_lower = category.lower()

            if "animal" in category_lower:
                color, symbol = "#B91C1C", "🔴"
            elif "plant" in category_lower:
                color, symbol = "#15803D", "🟢"
            else:
                color, symbol = "#B45309", "🟠"

            # Escape label text before inserting it into HTML.
            import html
            ingredient_name = html.escape(str(item.get("Ingredient", "")))
            explanation = html.escape(str(item.get("Explanation", "")))
            safe_category = html.escape(category)

            st.markdown(
                f"""
                <div class="ingredient-card">
                    <h4>{symbol} {ingredient_name}</h4>
                    <p><strong>Classification:</strong>
                    <span style="color:{color};font-weight:bold;">{safe_category}</span></p>
                    <p><strong>Explanation:</strong> {explanation}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### 📥 Download Your Report")
        csv_data = results_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Download CSV Report",
            data=csv_data,
            file_name="ingredient_analysis_report.csv",
            mime="text/csv"
        )
        st.caption("Results are estimates and may not confirm the actual source used by a manufacturer.")


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif page == "About":

    st.subheader("About This Project")

    st.write("""
    The Ingredient Origin Detection System helps users understand
    the possible origin of ingredients printed on personal care
    product labels.

    The application uses Optical Character Recognition (OCR) to
    extract ingredient names from images. It then compares those
    names against ingredient reference information and provides
    an estimated classification.

    **Main technologies**

    - Python
    - Streamlit
    - EasyOCR
    - OpenCV
    - Pandas
    """)

    st.warning(
        "This tool is intended for informational purposes only. "
        "It cannot guarantee an ingredient's actual source. "
        "For confirmation, contact the product manufacturer."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    Ingredient Origin Detector | ML-Based Ingredient Analysis
</div>
""", unsafe_allow_html=True)