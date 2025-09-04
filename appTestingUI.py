import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from peft import PeftModel
import openai
import os
from dotenv import load_dotenv

load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="CommentSense: AI Comment Analysis",
    page_icon="💡",
    layout="wide"
)

# --- App Title and Description ---
st.title("💡 CommentSense: AI-Powered Comment Intelligence")
st.markdown("""
Welcome to **CommentSense**, your tool for analyzing customer comments at scale. 
This prototype uses a three-stage AI pipeline:
1.  **Relevance Classification:** Determines if a comment is product-related.
2.  **Category Classification:** Assigns a product category (e.g., Skincare, Makeup).
3.  **Insight Generation:** Uses a Large Language Model (LLM) to extract actionable pain points and suggestions.
""")

# --- Sidebar for API Key Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("This app uses fine-tuned DistilBERT models for classification and OpenAI's GPT for insight generation.")
    st.markdown("---")
    st.warning("Your OpenAI API key is loaded from the `.env` file in the project directory.")

# --- Caching: Load Models Only Once ---
@st.cache_resource
def load_models():
    """
    Loads both fine-tuned DistilBERT models with their LoRA adapters from the Hugging Face Hub.
    Using st.cache_resource ensures this heavy operation runs only once.
    """
    base_model_name = "distilbert-base-uncased"
    hub_id_stage1 = "junmeng-sf/distilbert-base-product-related"
    hub_id_stage2 = "junmeng-sf/distilbert-base-category-classifier"
    
    # --- Load Stage 1 Model (Product Relevance) ---
    base_model_s1 = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        id2label={0: "Not Product Related", 1: "Product Related"}
    )
    model_s1 = PeftModel.from_pretrained(base_model_s1, hub_id_stage1)
    tokenizer_s1 = AutoTokenizer.from_pretrained(hub_id_stage1)
    classifier_stage1 = pipeline(
        "text-classification", model=model_s1, tokenizer=tokenizer_s1, device=0 if torch.cuda.is_available() else -1
    )

    # --- Load Stage 2 Model (Category Classification) ---
    id2label_stage2 = {0: 'makeup', 1: 'haircare', 2: 'skincare', 3: 'fragrance', 4: 'haircolor'}
    base_model_s2 = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=len(id2label_stage2),
        id2label=id2label_stage2
    )
    model_s2 = PeftModel.from_pretrained(base_model_s2, hub_id_stage2)
    tokenizer_s2 = AutoTokenizer.from_pretrained(hub_id_stage2)
    classifier_stage2 = pipeline(
        "text-classification", model=model_s2, tokenizer=tokenizer_s2, device=0 if torch.cuda.is_available() else -1
    )

    return classifier_stage1, classifier_stage2

# --- Function for LLM Insight Generation ---
## CHANGED ##: Simplified the function to use the passed api_key argument.
def get_llm_insight(comment, category, api_key):
    """
    Calls the OpenAI API to generate insights based on the comment and its category.
    """
    # The check now correctly uses the api_key passed as an argument
    if not api_key:
        # Improved error message for clarity
        st.error("OpenAI API key is not found. Please ensure it is set in your .env file.")
        return None
    
    client = openai.OpenAI(api_key=api_key)

    system_prompt = """You are a world-class Principal Analyst at a leading beauty corporation. Your mission is to transform a single customer comment into a strategic, actionable business memo. 
    
    Your analysis must be sharp, concise, and grounded strictly in the user's comment. Do not give vague suggestions like "improve the product." Be specific.
    
    You must structure your response in the following markdown format:

    **One-Line Summary:** A single sentence that captures the absolute core of the feedback.

    **Sentiment Analysis:** Describe the user's tone and emotion (e.g., "Frustrated with product longevity," "Pleased with performance but disappointed in packaging," "Seeking product information").

    **Core Insight & Pain Point:** Clearly identify the central issue or praise. What is the user's primary problem, question, or compliment?

    **Potential Business Impact:** If this single comment represents a wider issue, what is the risk or opportunity? (e.g., "Risk of negative word-of-mouth," "Opportunity for a new marketing claim," "Potential for customer churn.")

    **Actionable Recommendation:** Provide a concrete, specific action that can be taken.
    
    **Target Department:** Identify the primary department that should own this action (e.g., "Product Development," "Marketing," "Packaging Design," "Customer Service").
    """

    user_prompt = f"""
    Analyze the following customer comment regarding our '{category}' product:

    **Comment:** "{comment}"

    ---
    Based *only* on the information in this comment, provide the following:

    ### Pain Point:
    - Briefly describe the main problem or frustration the user is expressing. If there is no clear pain point, state "None detected."

    ### Actionable Suggestion:
    - Provide a specific, concrete suggestion for the product development or marketing team to address this feedback. If no action is needed, state "None."
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Changed to a more common and accessible model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=250,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred with the OpenAI API: {e}")
        return None

# --- Main Application Logic ---

# Load the models using the cached function
try:
    with st.spinner("Loading AI models... This might take a moment on first run."):
        classifier_stage1, classifier_stage2 = load_models()
except Exception as e:
    st.error(f"Could not load models. Please check your internet connection and Hugging Face Hub access. Error: {e}")
    st.stop()

## CHANGED ##: Load the OpenAI API key here so it's available for the check below.
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- UI for Single Comment Analysis ---
st.header("🔬 Analyze a Single Comment")
user_comment = st.text_area("Enter a customer comment below:", height=100, placeholder="e.g., 'This foundation is great, but the pump on the bottle broke after one week!'")

if st.button("Analyze Comment"):
    if not user_comment:
        st.warning("Please enter a comment to analyze.")
    # This check will now work correctly
    elif not openai_api_key:
        st.error("OpenAI API Key not found. Please add it to your .env file to generate insights.")
    else:
        with st.spinner("Running AI analysis..."):
            # Stage 1: Relevance Classification
            relevance_result = classifier_stage1(user_comment)[0]
            is_related = relevance_result['label'] == 'Product Related'
            relevance_score = relevance_result['score']

            st.subheader("Analysis Results")
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="**Stage 1: Is it Product Related?**",
                    value="Yes" if is_related else "No",
                    help=f"Confidence Score: {relevance_score:.2%}"
                )

            if is_related:
                # Stage 2: Category Classification
                category_result = classifier_stage2(user_comment)[0]
                category = category_result['label'].capitalize()
                category_score = category_result['score']
                
                with col2:
                    st.metric(
                        label="**Stage 2: Product Category**",
                        value=category,
                        help=f"Confidence Score: {category_score:.2%}"
                    )
                
                # Stage 3: LLM Insight Generation
            #     with st.spinner("Generating actionable insights with GPT..."):
            #         llm_insight = get_llm_insight(user_comment, category, openai_api_key)
                
            #     st.markdown("---")
            #     st.subheader("Stage 3: Actionable Insights")
            #     if llm_insight:
            #         st.markdown(llm_insight)
            # else:
            #     with col2:
            #         st.metric(label="**Stage 2: Product Category**", value="N/A")
            #     st.info("Comment is not product-related, so no category or insight is generated.")