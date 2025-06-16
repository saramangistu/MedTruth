# 🧬 MedTruth: Detecting Medical Misinformation on Social Media

This project aims to detect and classify **true vs. false health-related claims** in the style of **social media posts** using **Natural Language Processing (NLP)** and **large language models (LLMs)**. The solution combines real-world labeled datasets with synthetically generated claims via GPT-4-Turbo to improve model generalization and robustness.

---

## 🎯 Goal

- Build a robust binary classifier to detect **false medical claims** on social media.
- Evaluate performance on both **real** and **synthetic** data to simulate noisy, casual online text.
- Leverage **LLMs** (BERT-based) and compare with **baseline models**.

---

## 🖼️ Visual Abstract

![Visual Abstract](visual_abstract.png)

---

## 📦 Datasets

### 🧪 Real datasets:
- [COVID19 Fake News Dataset](https://www.kaggle.com/datasets/elvinagammed/covid19-fake-news-dataset-nlp)
- [PubHealth Dataset](https://www.kaggle.com/datasets/ersindemirel/pubhealthdataset)
- [HLR/Misinformation-Detection](https://github.com/HLR/Misinformation-Detection)

Only samples labeled as `True` or `False` were used.

### 🧠 Synthetic dataset:
- 2,300+ **synthetic false claims** were generated using **GPT-4-Turbo** via Azure OpenAI.
- Prompts were primed with COVID-related false claims to guide generation of realistic, misleading, and casual health misinformation across diverse topics.

---

## 🛠️ Models

### 🔹 Baseline Models
- **TF-IDF + Naive Bayes**
- **TF-IDF + Logistic Regression**

### 🔹 Transformer-Based Models
- **BERT**
- **BioBERT**
- **RoBERTa**

Trained using HuggingFace `Trainer` on the merged (real + synthetic) dataset.

---

## 📈 Evaluation

All models were evaluated using:

- **Accuracy**, **Precision**, **Recall**, **F1-score**
- **Confusion Matrix**
- Evaluations were performed on a stratified test split from the combined dataset.

**Note:** Baseline models were trained on real data only. Advanced models were trained on **real + synthetic data**.

---

## 🗂️ Folder Structure

'''

MedTruth/
├── notebooks/
│   ├── 1_Synthetic_claim_generation_and_scoring.ipynb
│   ├── 2_Baseline_models.ipynb
│   └── 3_Advanced_models_BERT_BioBERT_RoBERTa.ipynb
│
├── data/
│   ├── dataset_claims_for_eval.csv
│   ├── dataset_final_GPTclaims.csv
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
│   ├── MedTruth_Proposal.pdf
│   ├── MedTruth_Interim.pdf
│   └── MedTruth_Final.pdf
│
├── visual_abstract.png
├── README.md
└── requirements.txt

'''

---

## 🤝 Credits
- GPT-4-Turbo API access via Azure OpenAI
- HuggingFace Transformers and Datasets
- Seaborn & Matplotlib for visualization
