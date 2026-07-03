import pandas as pd
import numpy as np
import re
import nltk
import joblib

from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# DOWNLOAD NLTK
# ==========================================

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ==========================================
# LOAD DATASET
# ==========================================

columns = [
    "target",
    "id",
    "date",
    "flag",
    "user",
    "text"
]

df = pd.read_csv(
    "data/training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    names=columns
)

print("=" * 60)
print("Dataset Loaded Successfully")
print(df.shape)
print("=" * 60)

# ==========================================
# SAMPLE DATA
# ==========================================

df = df.sample(
    n=100000,
    random_state=42
).reset_index(drop=True)

print("Sample Size:", len(df))

# ==========================================
# TARGET LABELS
# ==========================================

df["sentiment"] = df["target"].map({
    0: "Negative",
    4: "Positive"
})

print(df["sentiment"].value_counts())

# ==========================================
# TEXT CLEANING
# ==========================================

stop_words = set(stopwords.words("english"))

custom_stopwords = {

    "im","ive","id","ill",

    "youre","youve","youll",

    "dont","didnt","doesnt",

    "cant","couldnt",

    "wouldnt","shouldnt",

    "isnt","arent",

    "wasnt","werent",

    "rt","http","https","www","com",

    "twitter","tweet",

    "today","tomorrow","yesterday",

    "one","time","day","night",

    "really","still","much",

    "know","think","going",

    "back","make","made",

    "take","look","see",

    "get","got","say","said",

    "yeah","yes","haha","lol",

    "thing","things","people",

    "work","amp"

}

all_stopwords = stop_words.union(custom_stopwords)

lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = str(text).lower()

    # Remove HTML entities
    text = text.replace("&quot;", " ")
    text = text.replace("quot", " ")
    text = text.replace("&amp;", " ")

    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags symbol but keep the word
    text = re.sub(r"#", "", text)

    # Keep only alphabets
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    words = text.split()

    words = [

        lemmatizer.lemmatize(word)

        for word in words

        if word not in all_stopwords
        and len(word) > 2

    ]

    return " ".join(words)

print("Cleaning Tweets...")

df["clean_text"] = df["text"].apply(clean_text)

print("Cleaning Completed")

# ==========================================
# TF-IDF FEATURES
# ==========================================

X = df["clean_text"]

y = df["sentiment"].map({
    "Negative":0,
    "Positive":1
})

vectorizer = TfidfVectorizer(

    max_features=5000,

    ngram_range=(1,2),

    min_df=5,

    max_df=0.90

)

X_vectorized = vectorizer.fit_transform(X)

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining Logistic Regression Model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# MODEL EVALUATION
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"Accuracy : {accuracy:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Negative",
            "Positive"
        ]
    )
)

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)

print("\nModel Saved Successfully")

# ==========================================
# EXPORT POWER BI DATA
# ==========================================

powerbi_df = df[
    [
        "text",
        "sentiment",
        "clean_text"
    ]
].copy()

powerbi_df.to_csv(
    "data/cleaned_sentiment.csv",
    index=False
)

metrics_df = pd.DataFrame({

    "Metric":[
        "Accuracy"
    ],

    "Value":[
        round(
            accuracy*100,
            2
        )
    ]

})

metrics_df.to_csv(
    "data/model_metrics.csv",
    index=False
)

print("Power BI Dataset Saved")

# ==========================================
# TOP POSITIVE KEYWORDS
# ==========================================

positive_stopwords = all_stopwords.union({

    "good",
    "great",
    "nice",
    "love",
    "like",
    "thanks",
    "thank",
    "happy",
    "hope",
    "morning",
    "quot",
    "twitter",
    "tweet"

})

positive_text = " ".join(
    df[df["sentiment"] == "Positive"]["clean_text"].astype(str)
)

positive_counter = Counter(

    word

    for word in positive_text.split()

    if word not in positive_stopwords
    and len(word) > 3
    and word.isalpha()

)

top_positive = pd.DataFrame(

    positive_counter.most_common(15),

    columns=["word","count"]

)

top_positive.to_csv(
    "data/top_positive_words.csv",
    index=False
)

print("\nTop Positive Keywords")
print(top_positive.head())

# ==========================================
# TOP NEGATIVE KEYWORDS
# ==========================================

negative_stopwords = all_stopwords.union({

    "feel",
    "home",
    "wish",
    "last",
    "well",
    "good",
    "great",
    "nice",
    "love",
    "like",
    "thanks",
    "thank",
    "happy",
    "hope",
    "morning",
    "quot",
    "twitter",
    "tweet",
    "today",
    "tomorrow",
    "watching",
    "watch",
    "friend",
    "friends",
    "people",
    "thing",
    "things",
    "going",
    "getting",
    "make",
    "made",
    "take",
    "look",
    "looking",
    "know",
    "think",
    "want",
    "need"

})

negative_text = " ".join(
    df[df["sentiment"] == "Negative"]["clean_text"].astype(str)
)

negative_counter = Counter(

    word

    for word in negative_text.split()

    if word not in negative_stopwords
    and len(word) > 3
    and word.isalpha()

)

top_negative = pd.DataFrame(

    negative_counter.most_common(15),

    columns=["word","count"]

)

top_negative.to_csv(
    "data/top_negative_words.csv",
    index=False
)

print("\nTop Negative Keywords")
print(top_negative.head())

# ==========================================
# OPTIONAL: REMOVE DUPLICATE WORDS
# ==========================================

top_positive = (
    top_positive
    .drop_duplicates(subset="word")
    .reset_index(drop=True)
)

top_negative = (
    top_negative
    .drop_duplicates(subset="word")
    .reset_index(drop=True)
)

# Save again after removing duplicates

top_positive.to_csv(
    "data/top_positive_words.csv",
    index=False
)

top_negative.to_csv(
    "data/top_negative_words.csv",
    index=False
)

# ==========================================
# DATA SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("DATA SUMMARY")
print("=" * 60)

print(f"Total Tweets       : {len(df):,}")

print(
    f"Positive Tweets    : {(df['sentiment']=='Positive').sum():,}"
)

print(
    f"Negative Tweets    : {(df['sentiment']=='Negative').sum():,}"
)

print(f"Vocabulary Size    : {len(vectorizer.vocabulary_):,}")

# ==========================================
# FILES GENERATED
# ==========================================

print("\nFiles Generated Successfully\n")

generated_files = [

    "sentiment_model.pkl",

    "tfidf_vectorizer.pkl",

    "data/cleaned_sentiment.csv",

    "data/model_metrics.csv",

    "data/top_positive_words.csv",

    "data/top_negative_words.csv"

]

for file in generated_files:
    print(f"✓ {file}")