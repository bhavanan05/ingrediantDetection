import streamlit as st
import pandas as pd
import joblib
import easyocr
import cv2
import numpy as np
import re
import os
import html
import logging
import contextlib
import io

from PIL import Image
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Animal-Derived Ingredient Detection",
    page_icon="🌿",
    layout="wide"
)

logging.getLogger("easyocr").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
    color: #202020;
}

[data-testid="stSidebar"] {
    background-color: #f0f7f1;
}

h1, h2, h3, h4 {
    color: #174d35 !important;
}

p, label {
    color: #202020;
}

.stButton > button,
.stDownloadButton > button {
    background-color: #237a50;
    color: white !important;
    border-radius: 8px;
    border: none;
    padding: 10px 22px;
    font-weight: bold;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #185c3a;
    color: white !important;
}

textarea {
    color: #202020 !important;
    background-color: #ffffff !important;
}

div[data-testid="stMetric"] {
    background-color: #f3faf4;
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #d8eadb;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# TEXT NORMALIZATION
# =====================================================

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_category_name(category):
    category = normalize_text(category)

    if (
        "animal" in category
        or "non vegan" in category
        or "nonvegan" in category
    ):
        return "Animal-Based"

    if "plant" in category or "vegan" in category:
        return "Plant-Based"

    return "Uncertain"


# =====================================================
# LOAD DATASETS SILENTLY
# =====================================================

@st.cache_data(show_spinner=False)
def load_ingredient_data():

    files = [
        "ingredients.csv",
        "datasetsssssfinal.csv",
        "datasets2.csv"
    ]

    dataframes = []

    for filename in files:

        if not os.path.exists(filename):
            continue

        try:
            try:
                df = pd.read_csv(
                    filename,
                    encoding="cp1252",
                    on_bad_lines="skip"
                )
            except Exception:
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
                (c for c in df.columns if "ingredient" in c),
                None
            )

            category_col = next(
                (
                    c for c in df.columns
                    if c in ["category", "classification", "label"]
                ),
                None
            )

            explanation_col = next(
                (
                    c for c in df.columns
                    if (
                        "explanation" in c
                        or "description" in c
                        or "reason" in c
                    )
                ),
                None
            )

            if ingredient_col and category_col:

                temp = pd.DataFrame()

                temp["ingredient"] = df[ingredient_col]
                temp["category"] = df[category_col]

                if explanation_col:
                    temp["explanation"] = df[explanation_col]
                else:
                    temp["explanation"] = ""

                temp = temp.dropna(
                    subset=["ingredient", "category"]
                )

                temp["ingredient"] = (
                    temp["ingredient"].astype(str).str.strip()
                )

                temp["category"] = (
                    temp["category"].astype(str).str.strip()
                )

                temp["explanation"] = (
                    temp["explanation"].fillna("").astype(str)
                )

                temp = temp[
                    (temp["ingredient"] != "")
                    & (temp["category"] != "")
                ]

                dataframes.append(temp)

        except Exception:
            continue

    if not dataframes:
        return pd.DataFrame(
            columns=["ingredient", "category", "explanation"]
        )

    final_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    final_df["_normalized"] = (
        final_df["ingredient"].map(normalize_text)
    )

    final_df = final_df.drop_duplicates(
        subset=["_normalized"],
        keep="first"
    )

    return final_df.drop(
        columns=["_normalized"]
    ).reset_index(drop=True)


# =====================================================
# BUILT-IN INGREDIENT INFORMATION
# =====================================================

BUILT_IN_INGREDIENTS = {

    "gelatin": (
        "Animal-Based",
        "Gelatin is commonly obtained from animal collagen."
    ),

    "collagen": (
        "Animal-Based",
        "Collagen is commonly sourced from animal tissues."
    ),

    "lanolin": (
        "Animal-Based",
        "Lanolin is obtained from sheep wool."
    ),

    "beeswax": (
        "Animal-Based",
        "Beeswax is produced by honeybees."
    ),

    "carmine": (
        "Animal-Based",
        "Carmine is a red pigment obtained from insects."
    ),

    "tallow": (
        "Animal-Based",
        "Tallow is rendered animal fat."
    ),

    "keratin": (
        "Animal-Based",
        "Keratin may be sourced from animal materials."
    ),

    "honey": (
        "Animal-Based",
        "Honey is produced by honeybees."
    ),

    "casein": (
        "Animal-Based",
        "Casein is a milk-derived protein."
    ),

    "shellac": (
        "Animal-Based",
        "Shellac is obtained from lac insects."
    ),

    "silk": (
        "Animal-Based",
        "Silk is produced by silkworms."
    ),

    "milk": (
        "Animal-Based",
        "Milk is an animal-derived ingredient."
    ),

    "aloe vera": (
        "Plant-Based",
        "Aloe vera is obtained from a plant."
    ),

    "coconut oil": (
        "Plant-Based",
        "Coconut oil is extracted from coconuts."
    ),

    "olive oil": (
        "Plant-Based",
        "Olive oil is extracted from olives."
    ),

    "shea butter": (
        "Plant-Based",
        "Shea butter is obtained from shea tree nuts."
    ),

    "sunflower oil": (
        "Plant-Based",
        "Sunflower oil is extracted from sunflower seeds."
    ),

    "soybean oil": (
        "Plant-Based",
        "Soybean oil is obtained from soybeans."
    ),

    "almond oil": (
        "Plant-Based",
        "Almond oil is obtained from almonds."
    ),

    "tea tree oil": (
        "Plant-Based",
        "Tea tree oil is extracted from tea tree leaves."
    ),

    "jojoba oil": (
        "Plant-Based",
        "Jojoba oil is obtained from jojoba seeds."
    ),

    "argan oil": (
        "Plant-Based",
        "Argan oil is obtained from argan tree kernels."
    )
}


# =====================================================
# LOAD MACHINE LEARNING MODELS
# =====================================================

def load_pickle_file(paths):

    for path in paths:

        if os.path.exists(path):

            try:
                return joblib.load(path)

            except Exception:
                continue

    return None


@st.cache_resource(show_spinner=False)
def load_ml_models():

    rf_model = load_pickle_file([
        "models/random_forest.pkl",
        "random_forest_classifier.pkl",
        "models/random_forest_classifier.pkl"
    ])

    svm_model = load_pickle_file([
        "models/svm.pkl",
        "svm.pkl"
    ])

    vectorizer = load_pickle_file([
        "models/tfidf_vectorizer.pkl",
        "models/vectorizer.pkl",
        "tfidf_vectorizer.pkl",
        "vectorizer.pkl"
    ])

    return rf_model, svm_model, vectorizer


# =====================================================
# MODEL PREDICTION
# =====================================================

def predict_model(model, vectorizer, ingredient):

    if model is None:
        return None

    try:

        if hasattr(model, "steps"):
            model_input = [ingredient]

        else:

            if vectorizer is None:
                return None

            model_input = vectorizer.transform([
                normalize_text(ingredient)
            ])

        prediction = model.predict(model_input)[0]

        return get_category_name(prediction)

    except Exception:
        return None


# =====================================================
# INGREDIENT CLASSIFICATION
# =====================================================

def classify_ingredient(ingredient, reference_data):

    normalized = normalize_text(ingredient)

    if normalized in BUILT_IN_INGREDIENTS:

        category, explanation = (
            BUILT_IN_INGREDIENTS[normalized]
        )

        return {
            "Ingredient": ingredient,
            "Category": category,
            "Explanation": explanation
        }

    if not reference_data.empty:

        matches = reference_data[
            reference_data["ingredient"].map(normalize_text)
            == normalized
        ]

        if not matches.empty:

            row = matches.iloc[0]

            explanation = str(
                row.get("explanation", "")
            ).strip()

            if not explanation or explanation.lower() == "nan":
                explanation = (
                    "Identified using available ingredient "
                    "reference information."
                )

            return {
                "Ingredient": ingredient,
                "Category": get_category_name(row["category"]),
                "Explanation": explanation
            }

    rf_model, svm_model, vectorizer = load_ml_models()

    rf_prediction = predict_model(
        rf_model,
        vectorizer,
        ingredient
    )

    svm_prediction = predict_model(
        svm_model,
        vectorizer,
        ingredient
    )

    if (
        rf_prediction is not None
        and svm_prediction is not None
    ):

        if rf_prediction == svm_prediction:
            final_category = rf_prediction
        else:
            final_category = "Uncertain"

    elif rf_prediction is not None:

        final_category = rf_prediction

    elif svm_prediction is not None:

        final_category = svm_prediction

    else:

        final_category = "Uncertain"

    explanations = {

        "Animal-Based": (
            "The available classification information indicates "
            "an animal-derived origin. Verify with the manufacturer."
        ),

        "Plant-Based": (
            "The available classification information indicates "
            "a plant-derived origin."
        ),

        "Uncertain": (
            "The ingredient source could not be confidently "
            "identified. It may vary by supplier."
        )
    }

    return {
        "Ingredient": ingredient,
        "Category": final_category,
        "Explanation": explanations[final_category]
    }


# =====================================================
# OCR READER
# =====================================================

@st.cache_resource(show_spinner=False)
def load_ocr_reader():

    output_buffer = io.StringIO()
    error_buffer = io.StringIO()

    with contextlib.redirect_stdout(output_buffer):

        with contextlib.redirect_stderr(error_buffer):

            reader = easyocr.Reader(
                ["en"],
                gpu=False,
                verbose=False
            )

    return reader


# =====================================================
# EXTRACT TEXT FROM IMAGE
# =====================================================

def extract_text_from_image(image):

    reader = load_ocr_reader()

    image_array = np.array(
        image.convert("RGB")
    )

    image_array = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    extracted_text = reader.readtext(
        image_array,
        detail=0,
        paragraph=True
    )

    return "\n".join(extracted_text)


# =====================================================
# SPLIT INGREDIENTS
# =====================================================

def split_ingredients(text):

    text = str(text).replace("\n", ",")

    text = re.sub(
        r"(?i)ingredients\s*:",
        "",
        text
    )

    text = re.sub(
        r"(?i)composition\s*:",
        "",
        text
    )

    parts = re.split(
        r"[,;]+",
        text
    )

    cleaned = []
    seen = set()

    for ingredient in parts:

        ingredient = ingredient.strip()

        ingredient = re.sub(
            r"^[\d\s.\-*•]+",
            "",
            ingredient
        )

        ingredient = re.sub(
            r"[.\s]+$",
            "",
            ingredient
        )

        if not ingredient:
            continue

        normalized = normalize_text(ingredient)

        if normalized and normalized not in seen:

            seen.add(normalized)
            cleaned.append(ingredient)

    return cleaned


# =====================================================
# PDF REPORT GENERATION
# =====================================================

def generate_pdf_report(results):

    if not results:
        raise ValueError("No analysis results available.")

    buffer = BytesIO()

    page_width, _ = A4

    left_margin = 12 * mm
    right_margin = 12 * mm

    available_width = (
        page_width - left_margin - right_margin
    )

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=right_margin,
        leftMargin=left_margin,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Ingredient Analysis Report",
        author="Animal-Derived Ingredient Detection"
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#174d35"),
        spaceAfter=8
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#237a50"),
        spaceBefore=12,
        spaceAfter=7
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        wordWrap="CJK"
    )

    header_style = ParagraphStyle(
        "ReportTableHeader",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
        alignment=TA_CENTER
    )

    def safe_paragraph(value, style=body_style):

        safe = html.escape(
            str(value if value is not None else "")
        )

        safe = safe.replace("\n", "<br/>")

        return Paragraph(safe, style)

    total = len(results)

    animal = sum(
        item.get("Category") == "Animal-Based"
        for item in results
    )

    plant = sum(
        item.get("Category") == "Plant-Based"
        for item in results
    )

    uncertain = total - animal - plant

    elements = [

        Paragraph(
            "Ingredient Analysis Report",
            title_style
        ),

        safe_paragraph(
            "Generated: "
            + datetime.now().strftime("%d-%m-%Y %I:%M %p")
        ),

        Spacer(1, 8),

        Paragraph(
            "Analysis Summary",
            heading_style
        )
    ]

    summary_data = [

        [
            "Total Ingredients",
            "Animal-Based",
            "Plant-Based",
            "Uncertain"
        ],

        [
            str(total),
            str(animal),
            str(plant),
            str(uncertain)
        ]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[available_width / 4] * 4,
        repeatRows=1
    )

    summary_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#237a50")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "BACKGROUND",
                (0, 1),
                (-1, -1),
                colors.HexColor("#f0f7f1")
            ),

            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    elements.extend([
        summary_table,
        Paragraph(
            "Ingredient Classification Details",
            heading_style
        )
    ])

    table_data = [[

        safe_paragraph(
            "Ingredient",
            header_style
        ),

        safe_paragraph(
            "Classification",
            header_style
        ),

        safe_paragraph(
            "Explanation",
            header_style
        )

    ]]

    for item in results:

        table_data.append([

            safe_paragraph(
                item.get("Ingredient", "")
            ),

            safe_paragraph(
                item.get("Category", "Uncertain")
            ),

            safe_paragraph(
                item.get("Explanation", "")
            )

        ])

    result_table = Table(
        table_data,
        colWidths=[
            available_width * 0.25,
            available_width * 0.22,
            available_width * 0.53
        ],
        repeatRows=1,
        hAlign="LEFT"
    )

    result_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#237a50")
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.45,
                colors.HexColor("#b7c8bc")
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),

            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#f3faf4")
                ]
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                6
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                6
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    elements.extend([

        result_table,

        Paragraph(
            "Important Note",
            heading_style
        ),

        safe_paragraph(
            "This report provides automated ingredient "
            "classification for informational purposes. "
            "Some ingredients may have different sources "
            "depending on their manufacturing process or supplier. "
            "Uncertain results require further verification "
            "with the product manufacturer."
        )
    ])

    document.build(elements)

    pdf_bytes = buffer.getvalue()

    buffer.close()

    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("Invalid PDF file generated.")

    return pdf_bytes


# =====================================================
# APPLICATION HEADER
# =====================================================

st.markdown("""
<h1 style="text-align:center;">
🌿 Animal-Derived Ingredient Detection
</h1>

<p style="text-align:center; font-size:16px;">
Identify animal-derived ingredients in everyday products
</p>

<hr>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🌿 Navigation")

page = st.sidebar.radio(
    "Select a Page",
    [
        "Home",
        "Ingredient Scanner",
        "About"
    ]
)


# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":

    st.markdown("""
    ## Welcome!

    Identify animal-derived ingredients in everyday
    personal-care products.

    ### What can this application do?

    - Upload ingredient-label images.
    - Capture labels using your camera.
    - Enter ingredient text manually.
    - Extract text from product labels.
    - View ingredient classifications.
    - Read ingredient explanations.
    - Download a PDF analysis report.
    """)

    st.success(
        "Start using the Ingredient Scanner from the sidebar."
    )


# =====================================================
# INGREDIENT SCANNER
# =====================================================

elif page == "Ingredient Scanner":

    st.markdown("## 🔍 Ingredient Scanner")

    st.write(
        "Upload a product label, capture an image, "
        "or manually enter ingredient information."
    )

    input_method = st.radio(
        "Choose how to provide ingredient information:",
        [
            "Upload Image",
            "Capture Image",
            "Enter Text Manually"
        ],
        horizontal=True,
        key="input_method"
    )

    uploaded_image = None

    # IMAGE UPLOAD

    if input_method == "Upload Image":

        uploaded_file = st.file_uploader(
            "Upload Ingredient Label",
            type=["png", "jpg", "jpeg", "webp"],
            key="ingredient_upload"
        )

        if uploaded_file is not None:

            try:
                uploaded_image = Image.open(
                    uploaded_file
                ).convert("RGB")

            except Exception:
                st.error("The selected image could not be opened.")

    # CAMERA CAPTURE

    elif input_method == "Capture Image":

        camera_image = st.camera_input(
            "Capture Ingredient Label",
            key="ingredient_camera"
        )

        if camera_image is not None:

            try:
                uploaded_image = Image.open(
                    camera_image
                ).convert("RGB")

            except Exception:
                st.error("The captured image could not be opened.")

    # IMAGE PREVIEW AND OCR

    if uploaded_image is not None:

        st.markdown("### 2. Image Preview")

        st.image(
            uploaded_image,
            caption="Ingredient Label Image",
            use_container_width=True
        )

        if st.button(
            "Extract Ingredient Text",
            use_container_width=True,
            key="extract_ocr"
        ):

            try:

                with st.status(
                    "Reading the label...",
                    expanded=False
                ) as status:

                    extracted_text = extract_text_from_image(
                        uploaded_image
                    )

                    status.update(
                        label="Text extraction complete.",
                        state="complete"
                    )

                st.session_state["extracted_text"] = extracted_text

                st.session_state.pop(
                    "analysis_results",
                    None
                )

                if extracted_text.strip():

                    st.success(
                        "Ingredient text extracted successfully!"
                    )

                else:

                    st.warning(
                        "No text found. Try a clearer image."
                    )

            except Exception as error:

                st.error(
                    "Unable to read the image. Please try another image."
                )

                st.caption(
                    f"Technical detail: {error}"
                )

    # MANUAL TEXT INPUT

    if input_method == "Enter Text Manually":

        st.markdown("### 2. Enter Ingredient Information")

        manual_text = st.text_area(
            "Enter ingredients separated by commas",
            placeholder=(
                "Example: Water, Glycerin, Aloe Vera, "
                "Collagen, Coconut Oil"
            ),
            height=150,
            key="manual_ingredient_text"
        )

        if manual_text.strip():

            st.session_state["extracted_text"] = manual_text

    # EXTRACTED TEXT

    if (
        "extracted_text" in st.session_state
        and st.session_state["extracted_text"].strip()
    ):

        st.markdown("### 3. Extracted Ingredient Text")

        edited_text = st.text_area(
            "Review or edit the extracted text",
            value=st.session_state["extracted_text"],
            height=150,
            key="editable_ingredient_text"
        )

        st.markdown("### 4. Analyze Ingredients")

        if st.button(
            "🔍 Analyze Ingredients",
            use_container_width=True,
            key="run_analysis"
        ):

            ingredients = split_ingredients(
                edited_text
            )

            if not ingredients:

                st.warning(
                    "Please provide valid ingredient information."
                )

            else:

                results = []

                total_ingredients = len(ingredients)

                # Load reference information once.
                reference_data = load_ingredient_data()

                # PROGRESS DISPLAY

                progress_bar = st.progress(0)

                progress_text = st.empty()

                current_item = st.empty()

                try:

                    for index, ingredient in enumerate(
                        ingredients,
                        start=1
                    ):

                        percent_before = int(
                            (index - 1) / total_ingredients * 100
                        )

                        progress_text.markdown(
                            f"**Analysis progress: "
                            f"{index - 1}/{total_ingredients} "
                            f"ingredients ({percent_before}%)**"
                        )

                        current_item.caption(
                            f"Currently analyzing: {ingredient}"
                        )

                        result = classify_ingredient(
                            ingredient,
                            reference_data
                        )

                        results.append(result)

                        percent = int(
                            index / total_ingredients * 100
                        )

                        progress_bar.progress(
                            percent,
                            text=f"Analyzing ingredients: {percent}%"
                        )

                        progress_text.markdown(
                            f"**Analysis progress: "
                            f"{index}/{total_ingredients} "
                            f"ingredients ({percent}%)**"
                        )

                    st.session_state["analysis_results"] = results

                    progress_text.success(
                        f"Analysis completed: {total_ingredients} "
                        f"of {total_ingredients} ingredients analyzed (100%)."
                    )

                    current_item.empty()

                    st.success(
                        "Ingredient analysis completed successfully!"
                    )

                except Exception as error:

                    progress_text.empty()
                    current_item.empty()

                    st.error(
                        "An error occurred during ingredient analysis."
                    )

                    st.caption(
                        f"Technical detail: {error}"
                    )

    # =================================================
    # CLASSIFICATION RESULTS
    # =================================================

    if "analysis_results" in st.session_state:

        results = st.session_state["analysis_results"]

        st.markdown("---")

        st.markdown("## 5. Classification Results")

        animal_count = sum(
            item["Category"] == "Animal-Based"
            for item in results
        )

        plant_count = sum(
            item["Category"] == "Plant-Based"
            for item in results
        )

        uncertain_count = sum(
            item["Category"] == "Uncertain"
            for item in results
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Ingredients",
            len(results)
        )

        col2.metric(
            "Animal-Based",
            animal_count
        )

        col3.metric(
            "Plant-Based",
            plant_count
        )

        col4.metric(
            "Uncertain",
            uncertain_count
        )

        # INDIVIDUAL RESULTS

        st.markdown("### Individual Ingredient Results")

        for item in results:

            category = item["Category"]

            if category == "Animal-Based":
                icon = "🔴"

            elif category == "Plant-Based":
                icon = "🟢"

            else:
                icon = "🟡"

            with st.container(border=True):

                st.markdown(
                    f"#### {icon} {html.escape(str(item['Ingredient']))}"
                )

                st.markdown(
                    f"**Classification:** {category}"
                )

                st.write(
                    f"**Explanation:** {item['Explanation']}"
                )

        # COMPLETE TABLE

        st.markdown("### Complete Analysis Table")

        display_df = pd.DataFrame([

            {
                "Ingredient": item["Ingredient"],
                "Classification": item["Category"],
                "Explanation": item["Explanation"]
            }

            for item in results

        ])

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # PDF DOWNLOAD
        # =================================================

        st.markdown("---")

        st.markdown("## 📥 Download Your Report")

        st.write(
            "Download your ingredient analysis as a PDF document."
        )

        try:

            pdf_data = generate_pdf_report(results)

            st.download_button(
                label="📄 Download PDF Analysis Report",
                data=pdf_data,
                file_name="Ingredient_Analysis_Report.pdf",
                mime="application/pdf",
                key="download_pdf_report",
                use_container_width=True
            )

        except Exception as error:

            st.error(
                "Unable to generate the PDF report."
            )

            st.caption(
                f"Technical detail: {type(error).__name__}: {error}"
            )

        # CLEAR ANALYSIS

        if st.button(
            "Clear Analysis",
            key="clear_analysis"
        ):

            for key in [
                "analysis_results",
                "extracted_text",
                "editable_ingredient_text"
            ]:

                st.session_state.pop(
                    key,
                    None
                )

            st.rerun()


# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "About":

    st.markdown("## About")

    st.write("""
    This application helps users identify ingredients
    that may be derived from animals in everyday
    personal-care products.

    Users can upload product-label images, capture
    labels using a camera, or manually enter ingredient
    information.

    The application extracts and analyzes ingredient
    names and displays their classifications with
    simple explanations.
    """)

    st.markdown("### Classification Categories")

    st.success(
        "🟢 Plant-Based: Ingredients obtained from plants."
    )

    st.error(
        "🔴 Animal-Based: Ingredients obtained from animals."
    )

    st.warning(
        "🟡 Uncertain: Ingredients whose source cannot "
        "be confidently identified."
    )

    st.info(
        "Some ingredients may have different sources "
        "depending on the manufacturer or supplier."
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div style="
    text-align:center;
    color:#777777;
    font-size:13px;
    padding:20px;
">
    🌿 Animal-Derived Ingredient Detection
</div>
""", unsafe_allow_html=True)