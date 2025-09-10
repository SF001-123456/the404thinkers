## 🧑‍💻 Data Preparation

Before **fine-tuning DistilBERT**, we performed several **data preparation steps** to ensure data quality and training efficiency.

### Step 1 – Dataset Integration
* Combined comments dataset and video dataset into a single source.
* Saved as multiple Parquet files with 10k rows each. Efficient storage, faster I/O, and easier handling of large datasets compared to CSV.

### Step 2 – Data Preprocessing
* Dropped unwanted columns (non-essential metadata).
* Removed missing values.
* Applied spam detection to filter out irrelevant comments, keeping only unspam data.
* Performed text preprocessing: token cleaning, lowercasing, removing special characters, etc.

### Step 3 – Keyword-based Filtering (Stage 1 Preparation)
* Used a curated set of beauty-related keywords (brands, general beauty terms, haircare/skincare specifics).
* Classified comments as Product-related or Not product-related.
* Only product-related comments were used for Stage 2 training.

---

## 🤖 Model Training

### Stage 1 – Binary Classification (Relevance Detection)
* Purpose: Detect whether a comment is product-related or not.
* Dataset: 68,580 rows with balanced labels.
* Output: Model filters irrelevant data before deeper categorization.

### Stage 2 – Multi-class Classification (Product Categorization)
* Purpose: Assign product-related comments into one of five categories:
    * Makeup
    * Skincare
    * Haircare
    * Haircolor
    * Fragrance
* Reasoning: The "PRODUCT-RELATED" word cloud showed dominant terms such as haircare, skincare, makeup, confirming these as the most relevant categories to focus on.
* Approach: Keyword-based classification was first applied to define labels before fine-tuning.

--- 

## ⏱️ Training Efficiency

* Both stages fine-tuned using DistilBERT with free T4 GPU on Google Colab.
* Training time kept under 1 hour for both models combined.
* Cost-free setup for prototyping purposes.

---

## ✅ Benefits

* Produces clean, structured, and labeled data for insights generation.
* Filters noise effectively (non-product comments removed).
* Improves classification accuracy and interpretability with the two-stage setup.
* Efficient training pipeline using lightweight models and free GPU resources.

---

## ⚠️ Limitations

* Keyword-based filtering may miss edge cases or nuanced product mentions.
* DistilBERT, while efficient, may not perform as strongly as larger transformer models.
* Retraining is needed if applying to new domains or languages.

---

## ⚠️ Disclaimer

* The model training process is provided as Jupyter Notebook (.ipynb) files and was executed in Google Colab with free T4 GPU.
* There is no guarantee that training will run successfully in a local environment without modification.



