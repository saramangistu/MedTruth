# 🧬 MedTruth: Detecting Medical Misinformation on Social Media

**MedTruth** is an NLP pipeline for detecting and classifying **true vs. false health-related claims** in the style of **social media posts**. This solution combines real-world labeled datasets with **synthetically generated claims** (via GPT-4-Turbo on Azure OpenAI) to improve model generalization, handle noisy language, and simulate real-world online health misinformation.

---

## 🎯 Goal

- Build a robust binary classifier to detect **false medical claims** in social media-like text.
- Improve model robustness by training on **both real and synthetic data**.
- Compare **baseline classifiers** to **fine-tuned transformer models**.
- Simulate noisy, casual, and emotional phrasing as often seen online.

---

## 🖼️ Visual Abstract

![Visual Abstract](visual_abstract.png)

---

## 📦 Datasets

### 🧪 Real Datasets
- [COVID19 Fake News Dataset](https://www.kaggle.com/datasets/elvinagammed/covid19-fake-news-dataset-nlp)
- [PubHealth Dataset](https://www.kaggle.com/datasets/ersindemirel/pubhealthdataset)
- [HLR/Misinformation-Detection](https://github.com/HLR/Misinformation-Detection)

Only samples explicitly labeled as `True` or `False` were retained for training and evaluation.

### 🧠 Synthetic Dataset
- Over **2,300 synthetic false claims** generated using **GPT-4-Turbo** (Azure OpenAI).
- Prompting was guided using COVID-style misinformation to produce diverse, informal, misleading, and plausible health-related posts across various topics.
- All synthetic claims were written in a social media style: casual tone, emojis, spelling variations, emotional phrasing.

---

## 🛠️ Models

### 🔹 Baseline Models (real data only)
- TF-IDF + **Naive Bayes**
- TF-IDF + **Logistic Regression**

### 🔹 Transformer-Based Models (real + synthetic)
- **BERT** (`bert-base-uncased`)
- **BioBERT** (`dmis-lab/biobert-base-cased-v1.1`)
- **RoBERTa** (`roberta-base`)

All transformer models were fine-tuned using HuggingFace `Trainer` API with default optimization settings.

---

## 📈 Evaluation

All models were evaluated using the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

> ⚠️ **Baselines** were trained on real data only.  
> ✅ **Transformer models** were trained on the **merged (real + synthetic)** dataset.  
> Stratified train-test splits ensured class balance for fair evaluation.

---

## 🧪 Results Summary

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Naive Bayes         | 0.77     | 0.78      | 0.75   | 0.75     |
| Logistic Regression | 0.78     | 0.78      | 0.76   | 0.77     |
| BERT                | 0.867    | 0.865     | 0.872  | 0.869    |
| BioBERT             | 0.860    | 0.862     | 0.859  | 0.860    |
| RoBERTa             | **0.875**| **0.874** | **0.879** | **0.876** |

> 🔹 RoBERTa achieved the best overall performance.  
> 🔸 Classical models showed lower recall, especially for the "False" class.

---

## 📁 Folder Structure

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

## 💻 Running the Code

You can run the notebooks either locally or in **Google Colab**.  
All code was developed and tested in Google Colab ✅

### 1. Clone the repository

```bash
git clone https://github.com/YourUsername/MedTruth.git
cd MedTruth
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

---

## 🧾 Requirements

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

- GPT-4-Turbo API via **Azure OpenAI**
- Model fine-tuning via **HuggingFace Transformers**
- Evaluation via **scikit-learn** and **evaluate**
- Visuals via **Matplotlib** and **Seaborn**
- Entire pipeline developed in **Google Colab**
