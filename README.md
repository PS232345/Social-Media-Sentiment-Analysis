# Social Media Sentiment Analysis

A machine learning project that classifies tweets as **Positive** or **Negative** using NLP preprocessing, TF-IDF vectorization, and Logistic Regression — with results visualized in an interactive Power BI dashboard.

<img width="1321" height="744" alt="Screenshot (97)" src="https://github.com/user-attachments/assets/2451d600-f6e0-4879-9ae5-5d7c721940e3" />


---

## 📌 Overview

This project applies Natural Language Processing (NLP) and Machine Learning to classify the sentiment of social media posts (tweets) as **Positive** or **Negative**. It uses the Sentiment140 dataset, performs extensive text cleaning, converts text to numerical features using TF-IDF, and trains a Logistic Regression model. Results are exported and visualized in a Power BI dashboard for easy interpretation.

---

## ✨ Features

- End-to-end NLP pipeline: cleaning, stopword removal, lemmatization
- TF-IDF feature extraction with unigrams and bigrams
- Logistic Regression classifier for sentiment prediction
- Model evaluation with accuracy, classification report, and confusion matrix
- Exportable datasets and metrics for BI/dashboard tools
- Top positive/negative keyword extraction
- Interactive Power BI dashboard for visualization

---

## 🗂️ Dataset

- **Source:** [Sentiment140](http://help.sentiment140.com/for-students) (1.6 million labeled tweets)
- **Sample used:** 100,000 tweets (randomly sampled for training efficiency)
- **Labels:** `0` = Negative, `4` = Positive (mapped to `Negative` / `Positive`)

> Note: The full dataset file (`training.1600000.processed.noemoticon.csv`) is not included in this repo due to size. Download it separately and place it in the `data/` folder.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| NLP | NLTK (stopwords, WordNet lemmatizer) |
| ML | scikit-learn (TF-IDF, Logistic Regression) |
| Data Handling | pandas, NumPy |
| Model Persistence | joblib |
| Visualization | Power BI |

---

## 🔄 Workflow

1. **Load Data** — Read raw tweet dataset with target labels
2. **Sample** — Take a manageable sample (100,000 rows) for faster processing
3. **Label Mapping** — Convert numeric targets to `Positive` / `Negative`
4. **Text Cleaning** — Lowercase, remove URLs/mentions/HTML entities/special characters, remove stopwords, lemmatize
5. **Feature Extraction** — TF-IDF vectorization (max 5,000 features, unigrams + bigrams)
6. **Train/Test Split** — 80/20 stratified split
7. **Model Training** — Logistic Regression classifier
8. **Evaluation** — Accuracy, classification report, confusion matrix
9. **Export** — Save model, vectorizer, and CSVs for Power BI
10. **Keyword Analysis** — Extract top positive/negative keywords

---

## 📊 Results

| Metric | Value |
|---|---|
| Total Tweets Analyzed | 100,000 |
| Positive Tweets | ~50.06% |
| Negative Tweets | ~49.94% |
| Model Accuracy | **80.56%** |

### Dashboard Preview

The Power BI dashboard includes:
- Total tweet count, positive/negative percentage, and model accuracy KPIs
- Sentiment distribution donut chart
- Sentiment breakdown bar chart
- Top positive and negative keyword bar charts

*(See `screenshots/` folder for dashboard and visualization images.)*

---

## 📁 Project Structure

```
Social Media Sentiment Analysis/
│
├── data/
│   ├── training.1600000.processed.noemoticon.csv   # raw dataset (not included)
│   ├── cleaned_sentiment.csv                        # cleaned + labeled output
│   ├── processed_sentiment.csv
│   ├── model_metrics.csv                             # accuracy metrics
│   ├── top_positive_words.csv
│   └── top_negative_words.csv
│
├── screenshots/
│   ├── sentiment_distribution.png
│   ├── confusion_matrix.png
│   ├── classification_report.png
│   ├── top_positive_words.png
│   ├── top_negative_words.png
│   └── ...
│
├── sentiment_analysis.py       # main pipeline script
├── trend_analysis.py           # additional trend analysis
├── sentiment_model.pkl         # trained model
├── tfidf_vectorizer.pkl        # fitted TF-IDF vectorizer
├── Social Media Sentiment Analysis.pbix   # Power BI dashboard
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/social-media-sentiment-analysis.git
cd social-media-sentiment-analysis

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```
pandas
numpy
nltk
scikit-learn
joblib
```

---

## ▶️ Usage

1. Download the [Sentiment140 dataset](http://help.sentiment140.com/for-students) and place it in the `data/` folder as `training.1600000.processed.noemoticon.csv`.
2. Run the main script:
   ```bash
   python sentiment_analysis.py
   ```
3. This generates:
   - `sentiment_model.pkl` and `tfidf_vectorizer.pkl` (trained model + vectorizer)
   - Cleaned dataset and keyword CSVs in `data/`
4. Open `Social Media Sentiment Analysis.pbix` in Power BI Desktop and refresh the data source to load the latest results.

### Optional Xquik Export Import

To analyze exported Xquik tweet data with the same training pipeline, convert a CSV
that includes `Tweet Text`, `Username`, and `Tweet Created At` headers into the
Sentiment140-style columns first:

```bash
python xquik_import.py xquik-export.csv data/xquik-sentiment140.csv
```

The generated file uses the same `target,id,date,flag,user,text` order expected by
the existing `sentiment_analysis.py` loader. Add labels before training if you use
the converted file as a supervised dataset.

---

## 🚀 Future Improvements

- Experiment with more advanced models (e.g., SVM, XGBoost, or transformer-based models like BERT)
- Add neutral sentiment class for finer-grained classification
- Hyperparameter tuning for TF-IDF and Logistic Regression
- Deploy as a web app for real-time sentiment prediction
- Expand keyword analysis with n-gram sentiment trends over time

---

## 👤 Author

**Prachi Sable**
B.E. Artificial Intelligence & Data Science, GSMCOE, Pune

---

## 📄 License

This project is licensed under the MIT License.
