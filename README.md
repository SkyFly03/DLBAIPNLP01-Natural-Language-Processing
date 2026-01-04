## ▸ Project Overview
- **Sentiment classification pipeline:** preprocessing, vectorization, modeling, and evaluation  
- **Progressive benchmarking:** small → medium → large datasets to assess generalization  
- **Model comparison:** classical ML approaches and transformer-based baseline (DistilBERT)

## ▸ Tech Stack
- **Language:** Python  
- **NLP & ML:** scikit-learn, spaCy, NLTK  
- **Deep Learning:** Hugging Face Transformers, PyTorch  
- **Data access:** Hugging Face Datasets, Kaggle  
- **Workflow:** JupyterLab

## ▸ Project Context
- **Academic origin:** university project  
- **Design goal:** structured NLP workflow reflecting applied text classification practice

---
---

# DLBAIPNLP01 – Natural Language Processing (NLP)

## Task 1: Sentiment Analysis on Movie Reviews ![image](https://github.com/user-attachments/assets/8f31d0ef-3732-4a45-b288-580a4c27c444)

This project builds and compares sentiment classification models using movie review datasets of increasing size and complexity.

---

## 3 Datasets:
The objective is to first develop and tune models on a smaller dataset, then test them on progressively larger datasets to measure generalization and robustness.
- **1). Rotten Tomatoes Critic Reviews** (Kaggle) – 250,000+ reviews (sampled for small-scale testing)  
- **2). SST-2 (Stanford Sentiment Treebank – binary)** (Hugging Face / GLUE) – ~70,000 labeled reviews (intermediate complexity)  
- **3). IMDb Dataset** (Hugging Face) – 50,000 labeled reviews (large-scale validation)

---

## 1. Project Objectives

- Preprocess movie review text using NLP techniques (cleaning, lemmatization, stop word removal)  
- Apply three vectorization methods: **CountVectorizer**, **TF-IDF**, and **word embeddings**  
- Train and compare classifiers: **Naive Bayes**, **SVM**, **Decision Trees**, **Random Forest**, and **Transformers**  
- Evaluate models using **Accuracy**, **F1-score**, and **Confusion Matrix**  
- Validate model robustness by testing across all three datasets  

---

## 2. Setup and Installation

This notebook is developed in **JupyterLab** using a Python virtual environment.  
The key libraries used include:

- `pandas`, `nltk`, `spacy` – for text preprocessing  
- `sklearn` – for vectorization and classical ML models  
- `transformers`, `torch` – for fine-tuning DistilBERT  
- `matplotlib`, `seaborn` – for visualization  
- `datasets`, `kaggle` – for loading and downloading datasets  

All packages are installed via `pip` in a custom environment.

### Run locally

```bash
git clone https://github.com/SkyFly03/DLBAIPNLP01-Natural-Language-Processing.git
cd DLBAIPNLP01-Natural-Language-Processing

pip install -r requirements.txt
jupyter lab
```
---

## 3. Dataset Overview and Analysis

### 3.1 Rotten Tomatoes Critic Reviews (Initial Testing)

- **Source**: [Kaggle – Rotten Tomatoes Dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)  
- **Sample**: ~6,000 reviews (5,644 after cleaning)  
- **Description**: Short, sentence-like critic reviews with binary sentiment labels  
- **Use**: Used as the initial testbed for model training and comparison

#### 3.1.1 Data Cleaning

- **Preprocessing**: Lowercasing/ Punctuation removal/ Tokenization/ Stop word removal (NLTK)/ Lemmatization (spaCy)
- **Mapped**: `Fresh` → `1`, `Rotten` → `0`

#### 3.1.2 Vectorization

- **TF-IDF**: Applied with a 5,000-token vocabulary limit  
- **CountVectorizer**: Also tested for comparison  
- **Word Embeddings**: Computed average spaCy word vectors per review

#### 3.1.3 Model Testing

- **Models trained and evaluated**: Naive Bayes/ SVM (LinearSVC)/ Logistic Regression/ Multi-Layer Perceptron (MLP)/ DistilBERT (transformer)
- **Evaluation metrics**: Accuracy/ F1 Score/ Classification Report/ Confusion Matrix (visualized using Seaborn)
  
![image](https://github.com/user-attachments/assets/08d0a05b-0149-497c-9498-88ff9a8acbc2)

#### 3.1.4 Summary

- TF-IDF generally outperformed CountVectorizer for classical models  
- Logistic Regression and SVM had strong performance  
- DistilBERT achieved the highest F1 score: **87.7%**  
- Naive Bayes showed positive class bias  
- Exploratory tests on ironic and neutral reviews showed limitations in model nuance

---

### 3.2 Stanford Sentiment Treebank (SST-2) (Intermediate Testing)

- **Source**: [Hugging Face – GLUE (sst2)](https://huggingface.co/datasets/nyu-mll/glue/viewer/sst2)
- **Size**: ~70,000 labeled short movie reviews (sentence-level)
- **Labels**: Binary sentiment (positive = 1, negative = 0)
- **Goal**: Evaluate sentence-level generalization on an intermediate-sized benchmark

#### 3.2.1 Data Cleaning

- Converted Hugging Face dataset into DataFrames  
- **Preprocessing**: Null and duplicate removal/ Lowercasing/ Lemmatization (spaCy)/ Stop word removal/ Created `clean_text` column

#### 3.2.2 Vectorization

- TF-IDF applied (5,000 features) for Naive Bayes, SVM, and Logistic Regression

#### 3.2.3 Model Testing

- **Models**: Naive Bayes/ SVM/ Logistic Regression/ DistilBERT (via Hugging Face pipeline)
- **Evaluation metrics**: Accuracy/ F1 Score/ Classification Report/ Confusion Matrix (visualized using Seaborn)

![image](https://github.com/user-attachments/assets/db1bb3b1-bb74-4f99-9ffb-37ef3cc4a520)

#### 3.2.4 Summary

- Logistic Regression achieved the best classical model F1 score: **80.6%**  
- DistilBERT achieved the highest F1 overall: **91.4%**  
- Classical models showed stable performance with TF-IDF  
- DistilBERT demonstrated superior handling of sentence-level input

---

### 3.3 IMDb Dataset (Final Evaluation)

- **Source**: [Hugging Face – IMDb](https://huggingface.co/datasets/imdb)  
- **Size**: 50,000 full-length user reviews  
- **Goal**: Test model scalability and performance on long-form text

#### 3.3.1 Data Cleaning

- Loaded positive/negative reviews from both train and test folders  
- Merged into a single DataFrame with binary labels  
- **Preprocessing**: Lowercasing/ Lemmatization (spaCy)/ Stop word removal/ Created `clean_text` column  
- Sampled 5,000 reviews for DistilBERT due to token limits

#### 3.3.2 Vectorization

- Applied TF-IDF vectorization (5,000 features)  
- Used as input for classical models

#### 3.3.3 Model Testing

- **Models**: Naive Bayes/ SVM/ Logistic Regression/ DistilBERT (evaluated on a 5,000-review subset)
- **Evaluation**: Accuracy and F1 Score/ Confusion Matrix and classification reports for all models

![image](https://github.com/user-attachments/assets/c347eb91-4cd9-4b72-807d-0663221f2db7)

#### 3.3.4 Summary

- Logistic Regression achieved the highest score: **89.1%**  
- DistilBERT close behind at **89.0%**  
- Naive Bayes performed well but was weaker on recall and balance  
- Classical models scaled well when paired with TF-IDF  

---

### 3.4 Overall Comparison

- Across all three datasets, DistilBERT achieved the highest F1 scores in most cases:

> - **Rotten Tomatoes**: DistilBERT (F1 = 87.7%) exceeded all classical models  
> - **SST-2**: DistilBERT (F1 = 91.4%) surpassed Logistic Regression (80.6%)  
> - **IMDb**: Logistic Regression (F1 = 89.1%) slightly outperformed DistilBERT (89.0%)

- Classical models like Logistic Regression and SVM delivered strong results when paired with TF-IDF  
- DistilBERT showed the best **overall consistency and generalization**, especially on short-to-medium inputs

![image](https://github.com/user-attachments/assets/31946f65-d4b3-4bcc-9aa8-70d92425d519)
**Conclusion**:  
DistilBERT is the most effective model in this study for binary sentiment classification across varied datasets and input lengths.

---

## Data & Attribution

This project uses publicly available datasets from Kaggle, Stanford University, and Hugging Face.
The work was conducted in an academic context and is not affiliated with or endorsed by the data providers.

