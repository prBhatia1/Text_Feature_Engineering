# Text Feature Engineering — Real-world Product Reviews

A complete Text Processing Pipeline that analyzes real user-generated Amazon product reviews and converts them into numerical features for machine learning models.

## Techniques Covered

| Technique | Output | Captures Frequency? | Captures Importance? |
|-----------|--------|---------------------|----------------------|
| **One Hot Encoding** | Binary (0/1) | No | No |
| **Bag of Words** | Integer counts | Yes | No |
| **TF-IDF** | Float weights (0–1) | Yes (TF) | Yes (IDF) |

## Pipeline Overview

1. **Dataset Collection** — Web scraping Amazon reviews using Selenium (+ pre-collected CSV fallback)
2. **Preprocessing** — Lowercase, tokenization, punctuation removal, stopword removal, lemmatization
3. **Vocabulary Creation** — Word frequency analysis with visualization
4. **Feature Engineering** — OHE, Bag of Words, TF-IDF with embedded vector representations
5. **Comparison Analysis** — Side-by-side comparison of all three approaches
6. **Sparse Matrix Analysis** — Shape, sparsity %, memory usage, and why sparse matrices are inefficient at scale
7. **Sentiment Classification** — Logistic Regression & Naive Bayes using BoW and TF-IDF features

## Dataset

`reviews_dataset.csv` — 100 Amazon product reviews (Haier Refrigerator) with binary sentiment labels (`positive` / `negative`).

## Setup

```bash
# Clone the repo
git clone https://github.com/prBhatia1/Text_Feature_Engineering.git
cd Text_Feature_Engineering

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- pandas, numpy, scikit-learn, nltk, matplotlib, selenium, webdriver-manager

## Key Takeaways

- **OHE** is simplest but loses word frequency information
- **BoW** captures frequency but treats all words equally
- **TF-IDF** highlights distinctive words by down-weighting common terms
- All three produce **high-dimensional sparse vectors** — modern dense embeddings (Word2Vec, BERT) are the scalable alternative
