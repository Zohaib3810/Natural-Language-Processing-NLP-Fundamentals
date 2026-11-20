# Natural-Language-Processing-NLP-Fundamentals
This repository contains practical implementations and analyses of core Natural Language Processing (NLP) techniques, split into four foundational questions. 

## Table of Contents
- [Question 1: Sentiment Analysis Pipeline](#question-1-sentiment-analysis-pipeline)
- [Question 2: Document Similarity with CountVectorizer](#question-2-document-similarity-with-countvectorizer)
- [Question 3: Vector Representations and Distance Metrics](#question-3-vector-representations-and-distance-metrics)
- [Question 4: Manual vs. Scikit-Learn TF-IDF Verification](#question-4-manual-vs-scikit-learn-tf-idf-verification)
- [Installation and Dependencies](#installation-and-dependencies)

---

## Question 1: Sentiment Analysis Pipeline

This section implements a complete text classification pipeline to classify statements as either Positive or Negative.

### Workflow
1. **Text Preprocessing**: Lowercasing, tokenization, and removing non-alphanumeric characters and NLTK english stopwords.
2. **Feature Extraction**: Converting the preprocessed text into TF-IDF vectors using `TfidfVectorizer` (limited to 1000 features).
3. **Data Splitting**: Dividing the dataset into training (80%) and testing (20%) sets with stratified sampling.
4. **Model Training**: Training both a Multinomial Naive Bayes model and a Logistic Regression model on the training data.
5. **Inference**: Testing predictions on a custom unseen phrase.
6. **Evaluation**: Comparing model performances on the test dataset using accuracy score metrics.

---

## Question 2: Document Similarity with CountVectorizer

This section explores how to measure structural similarities between texts using a custom cosine similarity algorithm.

### Workflow
1. **Count Matrix Construction**: Vectorizing five sample documents using `CountVectorizer` to build a baseline matrix.
2. **Manual Cosine Similarity**: Implementing a custom similarity function calculating the dot product divided by the magnitude of the vectors.
3. **Similarity Matrix**: Constructing a complete pairwise document similarity matrix mapped to a Pandas DataFrame.
4. **Analysis**: Extracting the most similar and least similar document pairs from the matrix.

---

## Question 3: Vector Representations and Distance Metrics

This section analyzes how binary and frequency tokenization techniques affect structural distance computations between documents.

### Workflow
1. **Vocabulary Building**: Constructing a unique, lowercase vocabulary set from the corpus manually.
2. **Vector Creation**:
    - **Binary Vectors**: Representing word presence/absence (1 or 0).
    - **Frequency Vectors**: Counting absolute word occurrences.
3. **Euclidean Distance**: Implementing a manual function to calculate pairwise Euclidean distances between document frequency vectors.
4. **Similarity Mapping**: Analyzing which documents are the closest (minimum distance) and which are the furthest (maximum distance) in the vector space.

---

## Question 4: Manual vs. Scikit-Learn TF-IDF Verification

An indepth analysis comparing manual formulations of Term Frequency-Inverse Document Frequency (TF-IDF) against Scikit-Learn's optimized module.

### Key Comparisons
- **Manual Calculation**: Implementing step-by-step term counts, document frequencies, and logarithmic scaling using standard math libraries.
- **Scikit-Learn Verification**: Using `TfidfVectorizer` with matching regular expression token patterns to generate equivalent feature space structures.
- **Discrepancy Analysis**: Highlighting why numerical matrix values differ (e.g., due to L2 normalization, sublinear TF scaling, and smooth-IDF features applied internally by Scikit-Learn).

---

## Installation and Dependencies

To run the notebook or script, ensure you have Python 3 installed along with the following libraries:

```bash
pip install numpy pandas scikit-learn nltk
```

You will also need to download the required NLTK datasets, which is handled dynamically inside the code:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
```
```
