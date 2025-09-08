import streamlit as st
import pandas as pd
import time
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from peft import PeftModel
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Configuration ---
FINAL_DATASET_DIR = Path("final_datasets")

# --- Page Setup ---
st.set_page_config(
    page_title="AI Comment Classifier & Dashboard", page_icon="🚀", layout="wide"
)


# ==============================================================================
# 1. SHARED FUNCTIONS (Models & Data Loading)
# ==============================================================================
@st.cache_resource
def load_models_for_testing():
    """
    Loads all three transformer models for the AI Sandbox page.
    Uses st.cache_resource to load them only once per session.
    """
    st.write(
        "Cache miss! Loading AI models for the Sandbox... (this may take a minute)")
    base_model_name = "distilbert-base-uncased"
    hub_id_stage1 = "junmeng-sf/distilbert-base-product-related"
    hub_id_stage2 = "junmeng-sf/distilbert-base-category-classifier"
    hub_id_stage3 = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    base_model_s1 = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=2, id2label={0: "Not Product Related", 1: "Product Related"}
    )
    model_s1 = PeftModel.from_pretrained(base_model_s1, hub_id_stage1)
    tokenizer_s1 = AutoTokenizer.from_pretrained(hub_id_stage1)
    classifier_stage1 = pipeline(
        "text-classification", model=model_s1, tokenizer=tokenizer_s1,
        device=0 if torch.cuda.is_available() else -1, truncation=True, max_length=512, torch_dtype=torch.float16
    )

    id2label_stage2 = {0: "makeup", 1: "haircare",
                       2: "skincare", 3: "fragrance", 4: "haircolor"}
    base_model_s2 = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=len(id2label_stage2), id2label=id2label_stage2
    )
    model_s2 = PeftModel.from_pretrained(base_model_s2, hub_id_stage2)
    tokenizer_s2 = AutoTokenizer.from_pretrained(hub_id_stage2)
    classifier_stage2 = pipeline(
        "text-classification", model=model_s2, tokenizer=tokenizer_s2,
        device=0 if torch.cuda.is_available() else -1, truncation=True, max_length=512, torch_dtype=torch.float16
    )

    model_s3 = AutoModelForSequenceClassification.from_pretrained(
        hub_id_stage3)
    tokenizer_s3 = AutoTokenizer.from_pretrained(hub_id_stage3)
    classifier_stage3 = pipeline(
        "sentiment-analysis", model=model_s3, tokenizer=tokenizer_s3,
        device=0 if torch.cuda.is_available() else -1, truncation=True, max_length=512, torch_dtype=torch.float16
    )
    return classifier_stage1, classifier_stage2, classifier_stage3


@st.cache_data
def load_and_clean_preprocessed_data(data_directory: Path):
    """
    Loads all pre-processed CSV files and cleans the resulting DataFrame.
    """
    csv_files = list(data_directory.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()

    df = pd.concat((pd.read_csv(file)
                   for file in csv_files), ignore_index=True)

    # --- Data Cleaning ---
    df.dropna(subset=['textOriginal'], inplace=True)
    df['Predicted_Category'] = df['Predicted_Category'].fillna(
        '[Not Applicable]')
    df['predicted_sentiment'] = df['predicted_sentiment'].fillna('others')
    numeric_cols = ['likeCount_comment', 'likeCount_video',
                    'favouriteCount', 'commentCount']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Ensure all required columns exist before calculation
    required_cols = ['likeCount_comment', 'likeCount_video', 'favouriteCount', 'commentCount']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0 # Add the column with zeros if missing

    # Re-calculate a basic share of engagement for plotting, if possible.
    denominator = df['likeCount_video'] + \
        df['favouriteCount'] + df['commentCount']
        
    df['comment_share_of_engagement'] = np.divide(
        df['likeCount_comment'], 
        denominator, 
        out=np.zeros_like(df['likeCount_comment'], dtype=float), 
        where=denominator != 0
    )

    return df


# ==============================================================================
# 2. PAGE RENDERING FUNCTIONS
# ==============================================================================

def render_dashboard_page(df: pd.DataFrame):
    """
    Displays all the charts and data tables for the dashboard view.
    """
    st.title("🚀 Marketing & R&D Insights Dashboard")
    st.markdown("---")

    product_df = df[df["isProductRelated"] == "Product Related"].copy()

    if product_df.empty:
        st.warning("No product-related comments found to build the dashboard.")
        return

    # --- High-Level Overview ---
    st.header("📈 High-Level Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Overall Sentiment")
        sentiment_counts = df["predicted_sentiment"].value_counts()
        color_map = {"positive": "#2ca02c", "neutral": "#ff7f0e",
                     "negative": "#d62728", "others": "#cccccc"}
        fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, hole=0.4, marker=dict(
            colors=[color_map.get(l, "#808080") for l in sentiment_counts.index]))])
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Comment Relevance Ratio")
        relevance_counts = df["isProductRelated"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=relevance_counts.index, values=relevance_counts.values, hole=0.4)])
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        st.subheader("Comment Volume by Category")
        category_counts = product_df[product_df["Predicted_Category"]
                                     != '[Not Applicable]']["Predicted_Category"].value_counts()
        fig = px.bar(category_counts, x=category_counts.index, y=category_counts.values, labels={
                     "x": "Category", "y": "Count"}, text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Sentiment & Engagement ---
    st.header("❤️ Sentiment & Engagement Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentiment Distribution by Category")
        sentiment_by_cat = product_df.groupby(
            ["Predicted_Category", "predicted_sentiment"]).size().reset_index(name="count")
        fig = px.bar(
            sentiment_by_cat, x="Predicted_Category", y="count", color="predicted_sentiment",
            title="Sentiment Breakdown for Each Product Category",
            labels={"count": "Number of Comments",
                    "Predicted_Category": "Category"},
            barmode="group", color_discrete_map={"positive": "#2ca02c", "neutral": "#ff7f0e", "negative": "#d62728"}
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Average Share of Engagement by Category")
        soe_by_category = product_df.groupby("Predicted_Category")[
            "comment_share_of_engagement"].mean().dropna().sort_values(ascending=False)
        fig = px.bar(
            soe_by_category, y=soe_by_category.index, x=soe_by_category.values,
            orientation="h", labels={"y": "Category", "x": "Avg. Share of Engagement"}, text_auto=".2%"
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    # --- Deep Dive Section ---
    st.markdown("---")
    st.header("🗣️ Deep Dive: Voice of the Customer")
    categories = sorted([cat for cat in product_df["Predicted_Category"].unique(
    ) if cat != '[Not Applicable]'])
    selected_category = st.selectbox(
        "Select a category to explore:", options=categories)

    view_type = st.radio(
        "Show me the...",
        ["Most Liked Comments", "Most Positive Comments", "Most Negative Comments"],
        horizontal=True,
    )

    if selected_category:
        category_df = product_df[product_df["Predicted_Category"]
                                 == selected_category]
        display_df = pd.DataFrame()

        if view_type == "Most Liked Comments":
            st.subheader(
                f"Top 5 Most Liked Comments for '{selected_category}'")
            display_df = category_df.nlargest(5, "likeCount_comment")
        elif view_type == "Most Positive Comments":
            st.subheader(
                f"Top 5 Most Liked Positive Comments for '{selected_category}'")
            display_df = category_df[category_df["predicted_sentiment"] == "positive"].nlargest(
                5, "likeCount_comment")
        else:  # Most Negative Comments
            st.subheader(
                f"Top 5 Most Liked Negative Comments for '{selected_category}'")
            display_df = category_df[category_df["predicted_sentiment"] == "negative"].nlargest(
                5, "likeCount_comment")

        if display_df.empty:
            st.info("No comments found for this selection.")
        else:
            for _, row in display_df.iterrows():
                with st.container(border=True):
                    st.markdown(
                        f"**👍 Likes: {int(row['likeCount_comment'])} | Sentiment: `{row['predicted_sentiment']}`**")
                    st.markdown(f"> {row['textOriginal']}")


def render_soe_calculator_page(df: pd.DataFrame):
    """
    Creates a what-if scenario planner for SOE based on user input.
    """
    st.title("🧮 SOE Calculator")
    st.markdown("""
    Use this tool to forecast how changes in engagement could impact your Share of Engagement. 
    Start by selecting a category to load its current data, then adjust the numbers to see a projected outcome.
    """)
    st.markdown("---")
    
    product_df = df[df["isProductRelated"] == "Product Related"].copy()
    if product_df.empty:
        st.warning("No product-related comments found to perform analysis.")
        return
        
    # --- Step 1: User Inputs ---
    st.subheader("Step 1: Set Your Scenario")
    input_col, results_col = st.columns(2)

    with input_col:
        # Load baseline data from a category
        available_categories = sorted([cat for cat in product_df['Predicted_Category'].unique() if cat != '[Not Applicable]'])
        base_category = st.selectbox(
            "Select a category to pre-fill data (optional):", 
            options=["None"] + available_categories
        )

        # Get current numbers based on selection
        current_comment_likes = 0
        current_video_likes = 0
        current_video_favs = 0
        current_video_comments = 0

        if base_category != "None":
            category_df = product_df[product_df['Predicted_Category'] == base_category]
            current_comment_likes = int(category_df['likeCount_comment'].sum())
            # For video stats, we take the sum as a proxy
            current_video_likes = int(category_df['likeCount_video'].sum())
            current_video_favs = int(category_df['favouriteCount'].sum())
            current_video_comments = int(category_df['commentCount'].sum())

        st.info("Enter your **projected** numbers below.")
        
        # User input fields for the "what-if" scenario
        proj_comment_likes = st.number_input(
            "Total Comment Likes:", 
            min_value=0, 
            value=current_comment_likes,
            step=1000,
            help="The total number of likes you expect on comments for this scenario."
        )
        proj_video_likes = st.number_input(
            "Total Video Likes:", 
            min_value=0, 
            value=current_video_likes,
            step=10000
        )
        proj_video_favs = st.number_input(
            "Total Video Favourites:", 
            min_value=0, 
            value=current_video_favs,
            step=1000
        )
        proj_video_comments = st.number_input(
            "Total Video Comments:", 
            min_value=0, 
            value=current_video_comments,
            step=500
        )

    # --- Step 2: Calculate and Display Results ---
    with results_col:
        st.subheader("Step 2: See the Impact")

        # --- Current State Calculation ---
        current_denominator = current_video_likes + current_video_favs + current_video_comments
        current_soe = (current_comment_likes / current_denominator) if current_denominator > 0 else 0
        
        # --- Projected State Calculation ---
        projected_denominator = proj_video_likes + proj_video_favs + proj_video_comments
        projected_soe = (proj_comment_likes / projected_denominator) if projected_denominator > 0 else 0
        
        # Calculate the change
        delta_soe = projected_soe - current_soe
        
        with st.container(border=True):
            st.markdown(f"#### Comparison for '{base_category}' Scenario")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown("##### Current State")
                st.metric(
                    label="Current SOE", 
                    value=f"{current_soe:.4%}",
                )
                st.text(f"Comment Likes: {current_comment_likes:,}")
                st.text(f"Video Engagements: {current_denominator:,}")
            
            with res_col2:
                st.markdown("##### Projected State")
                st.metric(
                    label="Projected SOE", 
                    value=f"{projected_soe:.4%}",
                    delta=f"{delta_soe:.4%} ({(delta_soe/current_soe):.2%})" if current_soe > 0 else "N/A"
                )
                st.text(f"Comment Likes: {proj_comment_likes:,}")
                st.text(f"Video Engagements: {projected_denominator:,}")
                
def render_sandbox_page():
    """
    Creates an interactive page for users to test the AI models with their own text.
    """
    st.title("🧪 AI Model Sandbox")
    st.markdown("Enter a comment below to see how our 3-stage AI pipeline classifies it in real-time.")
    st.markdown("---")

    with st.spinner("Warming up the AI models..."):
        classifier_s1, classifier_s2, classifier_s3 = load_models_for_testing()

    user_input = st.text_area("Enter a comment here:", "This new foundation is amazing, it lasts all day!", height=100)

    if st.button("Classify Comment", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text to classify.")
        else:
            with st.spinner("Analyzing..."):
                st.subheader("AI Classification Results")
                cols = st.columns(3)
                relevance_result = classifier_s1(user_input)[0]
                with cols[0]:
                    st.metric("1. Is it Product Related?", relevance_result['label'])
                    st.caption(f"Confidence: {relevance_result['score']:.2%}")
                if relevance_result['label'] == 'Product Related':
                    category_result = classifier_s2(user_input)[0]
                    with cols[1]:
                        st.metric("2. Product Category", category_result['label'].title())
                        st.caption(f"Confidence: {category_result['score']:.2%}")
                    sentiment_result = classifier_s3(user_input)[0]
                    with cols[2]:
                        st.metric("3. Sentiment", sentiment_result['label'].title())
                        st.caption(f"Confidence: {sentiment_result['score']:.2%}")
                else:
                    st.info("Since the comment is not product-related, further classification is skipped.", icon="ℹ️")


def render_data_explorer_page(df: pd.DataFrame):
    """
    Displays the full DataFrame with interactive filters.
    """
    st.title("📊 Data Explorer")
    st.markdown("View, filter, sort, and download the entire cleaned dataset.")
    st.markdown("---")

    st.subheader("Filter Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        relevance_filter = st.multiselect("Filter by Relevance:", options=df["isProductRelated"].unique(), default=df["isProductRelated"].unique())
    with col2:
        category_filter = st.multiselect("Filter by Category:", options=sorted(df["Predicted_Category"].unique()), default=sorted(df["Predicted_Category"].unique()))
    with col3:
        sentiment_filter = st.multiselect("Filter by Sentiment:", options=df["predicted_sentiment"].unique(), default=df["predicted_sentiment"].unique())

    mask = (df["isProductRelated"].isin(relevance_filter) & df["Predicted_Category"].isin(category_filter) & df["predicted_sentiment"].isin(sentiment_filter))
    filtered_df = df[mask]

    st.markdown("---")
    st.markdown(f"**Displaying {len(filtered_df):,} of {len(df):,} rows**")
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Filtered Data as CSV", data=csv_data, file_name='filtered_comment_data.csv', mime='text/csv')

    st.dataframe(filtered_df, use_container_width=True)


# ==============================================================================
# 4. MAIN APP CONTROLLER
# ==============================================================================

st.sidebar.title("Navigation")
page_options = [
    "📈 Dashboard", 
    "🧮 SOE Calculator",
    "📊 Data Explorer", 
    "🧪 AI Model Sandbox"
]
page = st.sidebar.radio("Go to", page_options)

# --- Load and process data based on the configuration ---
if page != "🧪 AI Model Sandbox":
    df = load_and_clean_preprocessed_data(FINAL_DATASET_DIR)
    if df.empty:
        st.header("Welcome!")
        st.warning("Could not load dashboard data. Please check the `final_datasets` directory and ensure it contains valid CSV files.")
        st.stop()

# --- Route to the selected page ---
if page == "📈 Dashboard":
    render_dashboard_page(df)
elif page == "🧮 SOE Calculator": 
    render_soe_calculator_page(df)
elif page == "📊 Data Explorer":
    render_data_explorer_page(df)
elif page == "🧪 AI Model Sandbox":
    render_sandbox_page()