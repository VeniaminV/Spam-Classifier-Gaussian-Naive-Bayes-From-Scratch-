# Spam Classifier — Gaussian Naive Bayes (From Scratch)

A from-scratch implementation of a Gaussian Naive Bayes classifier in pure Python (no ML libraries) to detect spam emails using the UCI Spambase dataset.

> **Course:** CS445 Machine Learning — Programming Assignment 2

---

## Overview

This classifier predicts whether an email is spam or not based on 57 numeric features (word/character frequencies and run-length statistics). The entire model — data loading, training, inference, and evaluation — is implemented without scikit-learn or any ML framework.

---

## Dataset

**UCI Spambase** — [https://doi.org/10.24432/C53G6X](https://doi.org/10.24432/C53G6X)

| Property | Value |
|---|---|
| Emails | 4,601 |
| Features | 57 (word freq., char freq., run-length stats) |
| Spam rate | ~39% |
| Label | 1 = spam, 0 = not spam |

> The dataset file (`spambase.data`) is not included in this repo. Download it from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/94/spambase) and place it in the project root.

---

## How It Works

**Gaussian Naive Bayes** models each feature as a Gaussian (normal) distribution, independently per class. To classify a new email:

1. Compute the log-prior for each class: `log P(class)`
2. For each of the 57 features, add the log-likelihood: `log P(feature | class)`
3. Predict the class with the higher total log-score

Log-space arithmetic is used throughout to prevent floating-point underflow when multiplying 57 small probabilities together.

---

## Project Structure

```
├── naive_bayes.py     # Main script — load, train, predict, evaluate
└── spambase.data      # Dataset (download separately, not included)
```

---

## Usage

**Requirements:** Python 3.x — standard library only (no pip installs needed)

```bash
# 1. Download the dataset from UCI and place it in the project root
# 2. Run the classifier
python naive_bayes.py
```

**Expected output:**
```
Loading spambase.data...
  4601 emails, 57 features, spam rate = 0.394

Step 1: splitting into train/test sets
  training: 2300 emails (spam rate: 0.394)
  test:     2301 emails (spam rate: 0.394)

Step 2: training the model
  P(not-spam) = 0.6061
  P(spam)     = 0.3939

Step 3: classifying 2301 test emails

Results:
  Accuracy  = 0.XXXX (XX.XX%)
  Precision = 0.XXXX
  Recall    = 0.XXXX

Confusion matrix:
  ...
```

---

## Implementation Details

| Component | Approach |
|---|---|
| **Data loading** | `csv` module, auto-detects header |
| **Train/test split** | Stratified 50/50 split (preserves spam ratio) |
| **Prior probability** | `log(n_class / n_total)` |
| **Likelihood** | Gaussian PDF in log-space |
| **Zero-std handling** | Floor std at `0.0001` to avoid division by zero |
| **Prediction** | Argmax of log-posterior across classes |

---

## Citation

Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999).
*Spambase* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C53G6X
