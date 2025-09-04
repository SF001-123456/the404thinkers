import streamlit as st
import pandas as pd
import time
from pathlib import Path
import io
import math
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from peft import PeftModel
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Configuration ---
DATA_DIR = Path("/workspace/datasets/processed_chunks/")
PROCESSED_DATA_PATH = DATA_DIR / "final_processed_data.parquet"
NUM_CHUNKS = 1
FILE_PREFIX = "processed_chunk_"
FILE_SUFFIX = ".parquet"
BATCH_SIZE = 128

# --- Page Setup ---
st.set_page_config(
    page_title="AI Comment Classifier & Dashboard", page_icon="🚀", layout="wide"
)


# ==============================================================================
# 1. MODEL & DATA LOADING FUNCTIONS
# ==============================================================================
@st.cache_resource
def load_models():
    _start_time = time.time()
    st.write(
        "Cache miss! Loading AI models for the first time... (this may take a minute)"
    )

    # --- Model Hub IDs ---
    base_model_name = "distilbert-base-uncased"
    hub_id_stage1 = "junmeng-sf/distilbert-base-product-related"
    hub_id_stage2 = "junmeng-sf/distilbert-base-category-classifier"
    hub_id_stage3 = (
        "cardiffnlp/twitter-roberta-base-sentiment-latest"  # New Sentiment Model
    )

    # --- Load Stage 1 (Relevance) ---
    base_model_s1 = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        id2label={0: "Not Product Related", 1: "Product Related"},
    )
    model_s1 = PeftModel.from_pretrained(base_model_s1, hub_id_stage1)
    tokenizer_s1 = AutoTokenizer.from_pretrained(hub_id_stage1)
    classifier_stage1 = pipeline(
        "text-classification",
        model=model_s1,
        tokenizer=tokenizer_s1,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=512,
        torch_dtype=torch.float16,
    )

    # --- Load Stage 2 (Category) ---
    id2label_stage2 = {
        0: "makeup",
        1: "haircare",
        2: "skincare",
        3: "fragrance",
        4: "haircolor",
    }
    base_model_s2 = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=len(id2label_stage2), id2label=id2label_stage2
    )
    model_s2 = PeftModel.from_pretrained(base_model_s2, hub_id_stage2)
    tokenizer_s2 = AutoTokenizer.from_pretrained(hub_id_stage2)
    classifier_stage2 = pipeline(
        "text-classification",
        model=model_s2,
        tokenizer=tokenizer_s2,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=512,
        torch_dtype=torch.float16,
    )

    # --- Load Stage 3 (Sentiment) ---
    tokenizer_s3 = AutoTokenizer.from_pretrained(hub_id_stage3)
    model_s3 = AutoModelForSequenceClassification.from_pretrained(hub_id_stage3 )
    classifier_stage3 = pipeline(
        "sentiment-analysis",
        model=model_s3,
        tokenizer=tokenizer_s3,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=512,
        torch_dtype=torch.float16,
    )

    duration = time.time() - _start_time
    return classifier_stage1, classifier_stage2, classifier_stage3, duration


@st.cache_data
def load_chunk(file_path: Path) -> pd.DataFrame:
    return pd.read_parquet(file_path)


# ==============================================================================
# 2. DASHBOARD RENDERING FUNCTION
# ==============================================================================
def render_dashboard(df: pd.DataFrame):
    st.title("🚀 Marketing & R&D Insights Dashboard")
    st.markdown("---")

    product_df = df[df["isProductRelated"] == "Product Related"].copy()

    # --- Row 1: High-Level Overview ---
    st.header("📈 High-Level Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Overall Sentiment")
        sentiment_counts = df["predicted_sentiment"].value_counts()
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    hole=0.4,
                    marker_colors={
                        "positive": "#2ca02c",
                        "neutral": "#ff7f0e",
                        "negative": "#d62728",
                    },
                )
            ]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Comment Relevance Ratio")
        relevance_counts = df["isProductRelated"].value_counts()
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=relevance_counts.index,
                    values=relevance_counts.values,
                    hole=0.4,
                )
            ]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.subheader("Comment Volume by Category")
        category_counts = product_df["Predicted_Category"].value_counts()
        fig = px.bar(
            category_counts,
            x=category_counts.index,
            y=category_counts.values,
            labels={"x": "Category", "y": "Count"},
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Row 2: Sentiment & Engagement ---
    st.header("❤️ Sentiment & Engagement Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution by Category")
        sentiment_by_cat = (
            product_df.groupby(["Predicted_Category", "predicted_sentiment"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            sentiment_by_cat,
            x="Predicted_Category",
            y="count",
            color="predicted_sentiment",
            title="Sentiment Breakdown for Each Product Category",
            labels={"count": "Number of Comments", "Predicted_Category": "Category"},
            barmode="group",
            color_discrete_map={
                "positive": "#2ca02c",
                "neutral": "#ff7f0e",
                "negative": "#d62728",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Average Share of Engagement by Category")
        soe_by_category = (
            product_df.groupby("Predicted_Category")["comment_share_of_engagement"]
            .mean()
            .dropna()
            .sort_values(ascending=False)
        )
        fig = px.bar(
            soe_by_category,
            y=soe_by_category.index,
            x=soe_by_category.values,
            orientation="h",
            labels={"y": "Category", "x": "Avg. Share of Engagement"},
            text_auto=".2%",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Row 3: Deep Dive into Comments ---
    st.header("💬 Deep Dive: Voice of the Customer")
    st.markdown("Explore the most impactful comments within each product category.")

    categories = product_df["Predicted_Category"].dropna().unique().tolist()
    selected_category = st.selectbox(
        "Select a category to explore:", options=categories
    )

    view_type = st.radio(
        "Show me the...",
        ["Most Liked Comments", "Most Positive Comments", "Most Negative Comments"],
        horizontal=True,
    )

    if selected_category:
        category_df = product_df[product_df["Predicted_Category"] == selected_category]

        if view_type == "Most Liked Comments":
            st.subheader(f"Top 5 Most Liked Comments for '{selected_category}'")
            display_df = category_df.nlargest(5, "likeCount_comment")
        elif view_type == "Most Positive Comments":
            st.subheader(
                f"Top 5 Most Liked Positive Comments for '{selected_category}'"
            )
            display_df = category_df[
                category_df["predicted_sentiment"] == "positive"
            ].nlargest(5, "likeCount_comment")
        else:  # Most Negative Comments
            st.subheader(
                f"Top 5 Most Liked Negative Comments for '{selected_category}'"
            )
            st.warning(
                "These comments represent customer pain points and opportunities for improvement.",
                icon="💡",
            )
            display_df = category_df[
                category_df["predicted_sentiment"] == "negative"
            ].nlargest(5, "likeCount_comment")

        if display_df.empty:
            st.info("No comments found for this selection.")
        else:
            for _, row in display_df.iterrows():
                with st.container(border=True):
                    st.markdown(
                        f"**👍 Likes: {row['likeCount_comment']} | Sentiment: `{row['predicted_sentiment']}`**"
                    )
                    st.markdown(f"> {row['textOriginal']}")


# ==============================================================================
# 3. MAIN APP LOGIC (Controller)
# ==============================================================================

if "page" not in st.session_state:
    st.session_state.page = "processing"

if st.session_state.page == "processing":
    st.title("🤖 AI-Powered Comment Analysis: Processing Pipeline")

# --- NEW: Check for pre-processed file ---
    if PROCESSED_DATA_PATH.exists():
        st.success("✅ Found pre-processed data file!")
        with st.spinner("Loading final data..."):
            st.session_state.processed_df = pd.read_parquet(PROCESSED_DATA_PATH)
        
        # You can optionally re-calculate timings or just show a message
        st.info("Skipping all processing steps and loading the final result from disk.")
        
        if st.button("🚀 Proceed to Dashboard", type="primary"):
            st.session_state.page = 'dashboard'
            st.rerun()

    # --- Run the full pipeline ONLY if the file doesn't exist ---
    else:
        # Load models
        classifier_stage1, classifier_stage2, classifier_stage3, model_load_duration = load_models()
        # ... (Your entire data loading and 3-stage inference logic goes here) ...
        # ...
        
        if 'processed_df' in st.session_state:
            # --- NEW: Save the final file ---
            with st.spinner("Saving final processed data to disk for future runs..."):
                st.session_state.processed_df.to_parquet(PROCESSED_DATA_PATH)
                
    st.session_state.time_model_loading = model_load_duration

    # --- Data Loading ---
    if "combined_df" not in st.session_state:
        # This block is unchanged
        st.subheader("Step 1: Loading Data")
        data_load_start_time = time.time()
        progress_bar = st.progress(0)
        status_text = st.empty()
        list_of_dfs = []
        for i in range(1, NUM_CHUNKS + 1):
            file_path = DATA_DIR / f"{FILE_PREFIX}{i}{FILE_SUFFIX}"
            status_text.text(f"Loading chunk {i}/{NUM_CHUNKS}")
            if file_path.exists():
                list_of_dfs.append(load_chunk(file_path))
            progress_bar.progress(i / NUM_CHUNKS)
        status_text.success("Combining...")
        if list_of_dfs:
            st.session_state.combined_df = pd.concat(list_of_dfs, ignore_index=True)
        else:
            st.session_state.combined_df = None
        st.session_state.time_data_loading = time.time() - data_load_start_time
        progress_bar.empty()
        status_text.empty()
    combined_df = st.session_state.get("combined_df")

    # --- AI Classification & Feature Engineering ---
    if combined_df is not None and "processed_df" not in st.session_state:
        st.subheader("Step 2: Running AI Classification")

        # Stage 1: Relevance
        s1_start_time = time.time()
        stage1_progress = st.progress(0)
        stage1_status = st.empty()
        results_stage1 = []
        texts_stage1 = combined_df["textOriginal"].astype(str).fillna("").tolist()
        total_batches_s1 = math.ceil(len(texts_stage1) / BATCH_SIZE)
        for i in range(total_batches_s1):
            batch_results = classifier_stage1(
                texts_stage1[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            )
            results_stage1.extend([res["label"] for res in batch_results])
            stage1_status.text(f"Stage 1 (Relevance): Batch {i+1}/{total_batches_s1}")
            stage1_progress.progress((i + 1) / total_batches_s1)
        combined_df["isProductRelated"] = results_stage1
        st.session_state.time_s1_inference = time.time() - s1_start_time
        stage1_status.success("Stage 1 Complete!")
        stage1_progress.empty()

        # Stage 2: Category
        s2_start_time = time.time()
        stage2_progress = st.progress(0)
        stage2_status = st.empty()
        product_related_df = combined_df[
            combined_df["isProductRelated"] == "Product Related"
        ].copy()
        if not product_related_df.empty:
            texts_stage2 = (
                product_related_df["textOriginal"].astype(str).fillna("").tolist()
            )
            total_batches_s2 = math.ceil(len(texts_stage2) / BATCH_SIZE)
            results_stage2 = []
            for i in range(total_batches_s2):
                batch_results = classifier_stage2(
                    texts_stage2[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
                )
                results_stage2.extend([res["label"] for res in batch_results])
                stage2_status.text(
                    f"Stage 2 (Category): Batch {i+1}/{total_batches_s2}"
                )
                stage2_progress.progress((i + 1) / total_batches_s2)
            combined_df.loc[product_related_df.index, "Predicted_Category"] = (
                results_stage2
            )
        else:
            combined_df["Predicted_Category"] = None
        st.session_state.time_s2_inference = time.time() - s2_start_time
        stage2_status.success("Stage 2 Complete!")
        stage2_progress.empty()

        # --- CORRECTED: Stage 3: Sentiment ---
        s3_start_time = time.time()
        stage3_progress = st.progress(0)
        stage3_status = st.empty()

        # 1. Initialize the new column with the default value 'others' for all rows.
        combined_df["predicted_sentiment"] = "others"

        # 2. Create a temporary DataFrame with only the product-related comments.
        #    This is the same subset of data used for Stage 2.
        product_related_df_s3 = combined_df[
            combined_df["isProductRelated"] == "Product Related"
        ].copy()

        # 3. Only run the model if there are product-related comments to process.
        if not product_related_df_s3.empty:
            stage3_status.text(
                f"Stage 3 (Sentiment): Analyzing {len(product_related_df_s3)} product-related comments..."
            )
            texts_stage3 = (
                product_related_df_s3["textOriginal"].astype(str).fillna("").tolist()
            )
            total_batches_s3 = math.ceil(len(texts_stage3) / BATCH_SIZE)
            results_stage3 = []

            for i in range(total_batches_s3):
                # Run the classifier on the batch of *product-related* texts
                batch_results = classifier_stage3(
                    texts_stage3[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
                )
                results_stage3.extend([res["label"].lower() for res in batch_results])
                stage3_status.text(f"Stage 3 (Sentiment): Batch {i+1}/{total_batches_s3}")
                stage3_progress.progress((i + 1) / total_batches_s3)

            # 4. Place the results back into the main DataFrame at the correct indices.
            #    Rows that were not product-related remain untouched and keep their 'others' value.
            combined_df.loc[product_related_df_s3.index, "predicted_sentiment"] = results_stage3
        else:
            # Handle the case where there are no product-related comments
            stage3_status.text("Stage 3 (Sentiment): No product-related comments to analyze.")
            stage3_progress.progress(1.0) # Mark as complete

        st.session_state.time_s3_inference = time.time() - s3_start_time
        stage3_status.success("Stage 3 Complete!")
        stage3_progress.empty()


        # Feature Engineering (This part is fine, no changes needed)
        st.subheader("Step 3: Final Feature Engineering")
        with st.spinner("Calculating Share of Engagement..."):
            denominator = (
                combined_df["likeCount_video"]
                + combined_df["favouriteCount"]
                + combined_df["commentCount"]
            )
            combined_df["comment_share_of_engagement"] = combined_df[
                "likeCount_comment"
            ] / denominator.replace(0, np.nan)
        st.success("Feature engineering complete!")
        st.session_state.processed_df = combined_df

    if "processed_df" in st.session_state:
        st.divider()
        st.header("✅ Processing Complete")
        st.subheader("⏱️ Performance Summary")
        # Update columns for 5 metrics
        cols = st.columns(5)
        total_time = 0
        metrics = {
            "Data Loading": "time_data_loading",
            "Model Loading": "time_model_loading",
            "Relevance (S1)": "time_s1_inference",
            "Category (S2)": "time_s2_inference",
            "Sentiment (S3)": "time_s3_inference",
        }
        for i, (label, key) in enumerate(metrics.items()):
            with cols[i]:
                time_val = st.session_state.get(key, 0)
                total_time += time_val
                st.metric(label, f"{time_val:.2f} s")
        st.success(f"**Total Processing Time: {total_time:.2f} seconds**")

        st.divider()
        if st.button("🚀 Proceed to Dashboard", type="primary"):
            st.session_state.page = "dashboard"
            st.rerun()

else:  # Dashboard View
    if "processed_df" in st.session_state:
        render_dashboard(st.session_state.processed_df)
    else:
        st.error("Processed data not found.")
        if st.button("Go back to Processing"):
            st.session_state.page = "processing"
            st.rerun()
