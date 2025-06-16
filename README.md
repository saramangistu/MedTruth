# 🧬 MedTruth: Detecting Medical Misinformation on Social Media

**MedTruth** is a full NLP pipeline for detecting **true vs. false health-related claims** written in the style of **social media posts**. The project combines real-world labeled datasets with **synthetic GPT-4-Turbo-generated claims** to simulate noisy, casual health misinformation, improving robustness and realism. We compare baseline and transformer-based models using HuggingFace and Colab.

---

## 📌 Table of Contents

- [🎯 Project Goal](#-project-goal)
- [🧠 Datasets](#-datasets)
- [🛠️ Models](#-models)
- [📈 Evaluation](#-evaluation)
- [📊 Results Summary](#-results-summary)
- [🧪 Key Insights](#-key-insights)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [📦 Requirements](#-requirements)
- [🤝 Credits](#-credits)

---

## 🎯 Project Goal

- Build a robust binary classifier for detecting **false medical claims** from social media-style texts.
- Leverage both **real-world datasets** and **LLM-generated synthetic claims** to enhance coverage.
- Evaluate performance across several models from classical ML to transformer-based LLMs.
- Explore robustness to casual phrasing, misinformation patterns, and low-resource domains.

---

## 🧠 Datasets

### 🧪 Real Datasets
- [COVID19 Fake News Dataset](https://www.kaggle.com/datasets/elvinagammed/covid19-fake-news-dataset-nlp)
- [PubHealth Dataset](https://www.kaggle.com/datasets/ersindemirel/pubhealthdataset)
- [HLR/Misinformation-Detection](https://github.com/HLR/Misinformation-Detection)

➡️ All datasets were filtered to include only samples labeled as `True` or `False`.

### 🧠 Synthetic Dataset
- Over **2,300 false claims** generated using **GPT-4-Turbo** via Azure OpenAI.
- Prompts were primed with COVID-style misinformation to generate **realistic**, **plausible**, and **diverse** false medical claims.
- Claims simulate social media language: informal, noisy, emotional, and sometimes misspelled.

---

## 🛠️ Models

### 🔸 Baseline Models (real data only)
- TF-IDF + **Naive Bayes**
- TF-IDF + **Logistic Regression**

### 🔹 Transformer Models (real + synthetic)
- **BERT** (`bert-base-uncased`)
- **BioBERT** (`dmis-lab/biobert-base-cased-v1.1`)
- **RoBERTa** (`roberta-base`)

All models were fine-tuned using HuggingFace `Trainer`.

---

## 📈 Evaluation

- **Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix.
- Models were evaluated on a **stratified test set**.
- All transformer models were trained on **combined (real + synthetic)** datasets.
- Baselines were trained on **real data only**.

---

## 📊 Results Summary

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Naive Bayes         | 0.77     | 0.78      | 0.75   | 0.75     |
| Logistic Regression | 0.78     | 0.78      | 0.76   | 0.77     |
| BERT                | 0.867    | 0.865     | 0.872  | 0.869    |
| BioBERT             | 0.860    | 0.862     | 0.859  | 0.860    |
| RoBERTa             | **0.875**| **0.874** | **0.879** | **0.876** |

---

## 🧪 Key Insights

- ✅ **RoBERTa** outperformed all other models across all metrics.
- 🔬 Synthetic GPT-4 claims enhanced generalization and robustness.
- ⚠️ Classical baselines underperformed, especially on identifying false claims.
- 🔁 Real + synthetic data proved more effective than real-only setups.

---

## 📁 Project Structure

```
MedTruth/
├── notebooks/
│   ├── Synthetic_claims_generation_and_scoring.ipynb
│   ├── Baseline_models.ipynb
│   └── Advanced_models_BERT_BioBERT_RoBERTa.ipynb
│
├── data/
│   ├── claims_for_eval.csv
│   ├── final_GPTclaims.csv
│   ├── dataset_final_baseline_data.csv
│   └── dataset_final_advanced_data.csv
│
├── results/
│   └── graphs/
│       ├── confusion_matrix_BERT.png
│       ├── confusion_matrix_BioBERT.png
│       ├── confusion_matrix_RoBERTa.png
│       ├── confusion_matrix_logistic.png
│       └── confusion_matrix_naive_bayes.png
│
├── presentations/
│   ├── MedTruth - Proposal Presentation.pdf
│   ├── MedTruth - Interim Presentation.pdf
│   └── MedTruth - Final Presentation.pdf
│
├── visual_abstract.png
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YourUsername/MedTruth.git
cd MedTruth
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

### 3. Run notebooks in Google Colab or locally

---

## 📦 Requirements

```
transformers==4.52.4
datasets>=3.6.0
evaluate>=0.4.3
scikit-learn>=1.4.2
pandas>=2.2.2
numpy>=1.26.4
matplotlib>=3.8.4
seaborn>=0.13.2
torch>=2.3.0
nltk>=3.9.1
spacy>=3.8.5
tiktoken>=0.6.0
kagglehub>=0.1.6
backoff>=2.2.1
openai>=1.30.5
```

---

## 🤝 Credits

- GPT-4-Turbo access via **Azure OpenAI**
- Transformer fine-tuning via **HuggingFace**
- Visualization with **Seaborn** and **Matplotlib**
- Entire pipeline built in **Google Colab**
