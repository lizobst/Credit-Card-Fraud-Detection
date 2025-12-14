# Credit Card Fraud Detection Using Deep Neural Networks


## Project Overview

This project evaluates the effectiveness of deep neural networks for detecting fraudulent credit card transactions. Using the Kaggle Credit Card Fraud Detection dataset, we developed and compared multiple machine learning approaches to address the challenge of identifying fraud in highly imbalanced transaction data.

## Problem Statement

Credit card fraud costs the financial industry billions of dollars annually. Traditional rule-based systems fail to adapt to evolving fraud tactics and generate excessive false positives that burden investigation teams. This research explores whether deep learning can provide a more effective solution for real-world fraud detection.

## Dataset

- **Source:** Kaggle Credit Card Fraud Detection Dataset
- **Size:** 284,807 European credit card transactions (September 2013)
- **Class Distribution:** 492 fraudulent (0.172%) vs. 284,315 legitimate (99.828%)
- **Features:** 30 total (28 PCA-transformed components + Time + Amount)

## Key Challenges

1. **Extreme Class Imbalance:** Only 0.172% of transactions are fraudulent, causing models to achieve high accuracy by predicting everything as legitimate
2. **Precision-Recall Trade-off:** Balancing the need to catch fraud (recall) while minimizing false alarms (precision)
3. **Anonymized Features:** Working with PCA-transformed data that limits interpretability
4. **Evaluation Metrics:** Traditional accuracy is misleading for imbalanced data

## Approach

### Deep Neural Network Architecture
- **Input Layer:** 30 features
- **Hidden Layers:** 64 → 32 → 16 neurons with ReLU activation
- **Dropout Regularization:** 30%, 30%, 20% to prevent overfitting
- **Output Layer:** Sigmoid activation for binary classification
- **Training:** Adam optimizer, binary cross-entropy loss, 20 epochs

### Addressing Class Imbalance
Applied SMOTE (Synthetic Minority Over-sampling Technique) to training data only, balancing classes to 50% while preserving the original imbalanced test set for realistic evaluation.

### Baseline Models
- **XGBoost:** Gradient boosting with class weight balancing
- **Logistic Regression:** Simple interpretable baseline with balanced class weights

## Results

### Model Performance Comparison

| Model | Precision | Recall | F1-Score | PR-AUC |
|-------|-----------|--------|----------|---------|
| **Deep Neural Network** | 83% | 72% | 0.768 | 0.744 |
| **XGBoost** | 82% | 84% | 0.802 | 0.811 |
| **Logistic Regression** | 6% | 87% | 0.11 | - |

### Key Findings

**Deep Neural Network:**
- Successfully identified 68 of 95 fraudulent transactions
- Generated only 19 false alarms
- Achieved highest precision (83%), minimizing wasted investigation resources

**XGBoost:**
- Best overall performance with superior balance between precision and recall
- Caught 75 of 95 frauds with 23 false positives
- Aligns with literature showing gradient boosting excels on tabular data

**Logistic Regression:**
- Highest recall (87%) but impractical precision (6%)
- Generated 1,386 false positives, overwhelming investigation teams
- ~25 fraud alerts per 1,000 transactions, only 1-2 genuine

## Business Context

Fraud detection requires balancing two objectives:
- **Catching fraud (recall):** Minimizes financial losses from missed fraudulent transactions
- **Minimizing false alarms (precision):** Reduces wasted investigation resources and customer frustration

The neural network's superior precision represents a critical advantage for production systems where investigation resources are limited and false alarms are costly.

## Analysis

While XGBoost outperformed the neural network on this dataset, deep learning shows significant promise for production fraud detection with:
- Larger-scale data (millions/billions of transactions)
- Temporal patterns (using RNN/LSTM architectures)
- Multi-modal data (text, images, geographic data)
- Complex non-linear feature interactions

The Kaggle dataset's 280K pre-processed transactions with anonymized features represents a simplified scenario. Production systems with rich, multi-dimensional data favor neural networks' representation learning capabilities.

## Conclusions

This research demonstrates that deep neural networks achieve competitive performance for credit card fraud detection, with the DNN's 83% precision. While tree-based methods remain pragmatic for many scenarios, neural networks represent a powerful fraud detection tool that will become increasingly valuable as data volume and complexity grow.

## Future Work

1. **RNN/LSTM Architectures:** Capture temporal transaction patterns and behavioral anomalies
2. **Ensemble Approaches:** Combine multiple network architectures for improved robustness
3. **Attention Mechanisms:** Focus on the most relevant features for each transaction
4. **Richer Datasets:** Train on larger datasets with additional features like merchant categories and geographic data
