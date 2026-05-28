# Hospital Review NLP Analyzer

A Python script that analyzes hospital/healthcare feedback using NLP to extract sentiment distribution and key positive aspects from patient reviews.

## What It Does

- Loads a CSV of patient reviews with feedback text, sentiment labels, and ratings
- Maps numeric ratings to sentiment categories (Positive / Neutral / Negative)
- Extracts top recurring keywords and keyphrases from positive reviews using **KeyBERT**
- Outputs a structured summary with average rating, sentiment breakdown, and top positive aspects

## Setup

```bash
# Clone the repo
git clone https://github.com/your-username/nlp.git
cd nlp

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Place your CSV file in the project directory
2. Open `nlp.py` and replace `"put_your_file_here.csv"` with your file name
3. Make sure your CSV has columns: `Feedback`, `Sentiment Label`, `Ratings`
4. Run the script:

```bash
python nlp.py
```

## Expected Output

```
Average Rating: 4.12

Sentiment Distribution (%):
Positive    65.30
Neutral     20.10
Negative    14.60

Top Positive Aspects:
- clean environment (8 mentions)
- friendly staff (7 mentions)
...
```

## Requirements

| Package | Version |
|---|---|
| pandas | ≥ 2.0.0 |
| numpy | ≥ 1.24.0 |
| scikit-learn | ≥ 1.3.0 |
| keybert | ≥ 0.8.0 |
| sentence-transformers | ≥ 2.2.2 |
| transformers | ≥ 4.36.0 |

## Sentiment Rating Scale

| Rating | Category |
|---|---|
| 4–5 | Positive |
| 3 | Neutral |
| 1–2 | Negative |
