# 🎙️ VoiceLens – AI Comment Classifier & Insights Dashboard

Developed by **The 404 Thinkers**

VoiceLens is a **Streamlit-powered web application** that uses **AI and interactive dashboards** to transform customer comments into actionable **marketing & R\&D insights**.

It includes:

* Multi-stage **AI classification pipeline** (Relevance → Product Category → Sentiment)
* Interactive **dashboard visualizations** with Plotly
* A **Share of Engagement (SOE) Calculator** for scenario planning
* Full **data explorer** with filtering and export
* AI Sandbox for **real-time comment classification**

---

## 📂 Project Structure

```
.
├── final_datasets/          # Pre-processed CSV datasets (input data)
├── app.py                   # Main Streamlit app
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## ⚙️ Installation

1. **Clone this repository**

```bash
git clone https://github.com/SF001-123456/the404thinkers.git
cd the404thinkers
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🛠️ Requirements

Main libraries used in this project:

* [Streamlit](https://streamlit.io/) – Web app framework
* [Transformers](https://huggingface.co/transformers/) – Pre-trained NLP models
* [PEFT](https://github.com/huggingface/peft) – Parameter-efficient fine-tuning
* [Torch](https://pytorch.org/) – Deep learning backend
* [Plotly](https://plotly.com/python/) – Interactive data visualization
* [Pandas](https://pandas.pydata.org/) – Data processing

Check `requirements.txt` for the full list.

---

## ▶️ Usage

Run the Streamlit app locally:

```bash
streamlit run app.py
```

Then open your browser at **[http://localhost:8501](http://localhost:8501)**.

---

## 📑 Features

### 1. **📈 Dashboard**

* High-level overview of **sentiment, relevance, and category trends**
* Category-level **engagement and sentiment analysis**
* Deep dive into **most liked, positive, and negative comments**

### 2. **🧮 SOE Calculator**

* Interactive tool to **simulate engagement scenarios**
* Projects **Share of Engagement (SOE)** based on user-defined inputs

### 3. **📊 Data Explorer**

* Filter, sort, and search through the **entire cleaned dataset**
* Export filtered results as **CSV**

### 4. **🧪 AI Model Sandbox**

* Real-time classification pipeline:

  1. **Relevance** – Product-related vs. Not
  2. **Category** – Makeup, Haircare, Skincare, etc.
  3. **Sentiment** – Positive, Neutral, Negative

---

## 📊 Example Workflow

1. Load **preprocessed CSVs** in `final_datasets/`
2. Navigate through the sidebar:

   * 📈 Dashboard → Visualize insights
   * 🧮 SOE Calculator → Plan scenarios
   * 📊 Data Explorer → Explore raw/filtered data
   * 🧪 AI Sandbox → Test AI models on new comments

---

## 📌 Notes

* GPU is recommended for faster model inference (uses CUDA if available).
* Ensure the `final_datasets/` folder contains valid CSVs with the required columns (`textOriginal`, `Predicted_Category`, `predicted_sentiment`, etc.).
* Models are loaded from Hugging Face Hub (`distilbert` + PEFT adapters + `twitter-roberta-base-sentiment`).

---

## 🛣️ Workflows

1. Prototype Workflow
   [Description to be inserted here]
   ![Prototype Workflow](prototype_workflow.png)
2. Production Workflow
   [Description to be inserted here]
   ![Production Workflow](production_workflow.png)

---

## 👥 Team

**The 404 Thinkers**
Creators of **VoiceLens**
