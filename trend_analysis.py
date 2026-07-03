import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import os

# ==============================
# Create Output Folder
# ==============================

os.makedirs("screenshots", exist_ok=True)

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("data/cleaned_sentiment.csv")

print("Dataset Loaded")
print(df.shape)

# ==============================
# Positive & Negative Tweets
# ==============================

positive_text = " ".join(
    df[df["sentiment"] == "Positive"]["clean_text"]
)

negative_text = " ".join(
    df[df["sentiment"] == "Negative"]["clean_text"]
)

# ==============================
# Positive Word Cloud
# ==============================

positive_wc = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(positive_text)

plt.figure(figsize=(12,6))
plt.imshow(positive_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Word Cloud")

plt.savefig(
    "screenshots/positive_wordcloud.png"
)

plt.show()

# ==============================
# Negative Word Cloud
# ==============================

negative_wc = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(negative_text)

plt.figure(figsize=(12,6))
plt.imshow(negative_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Word Cloud")

plt.savefig(
    "screenshots/negative_wordcloud.png"
)

plt.show()

# ==============================
# Sentiment Distribution
# ==============================

sentiment_counts = df["sentiment"].value_counts()

plt.figure(figsize=(8,5))

sentiment_counts.plot(
    kind="bar"
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "screenshots/sentiment_distribution.png"
)

plt.show()

# ==============================
# Top Positive Words
# ==============================

positive_words = positive_text.split()

top_positive = Counter(
    positive_words
).most_common(10)

positive_df = pd.DataFrame(
    top_positive,
    columns=["Word","Count"]
)

plt.figure(figsize=(10,5))

plt.bar(
    positive_df["Word"],
    positive_df["Count"]
)

plt.title("Top Positive Words")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "screenshots/top_positive_words.png"
)

plt.show()

# ==============================
# Top Negative Words
# ==============================

negative_words = negative_text.split()

top_negative = Counter(
    negative_words
).most_common(10)

negative_df = pd.DataFrame(
    top_negative,
    columns=["Word","Count"]
)

plt.figure(figsize=(10,5))

plt.bar(
    negative_df["Word"],
    negative_df["Count"]
)

plt.title("Top Negative Words")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "screenshots/top_negative_words.png"
)

plt.show()

print("\nAll Visualizations Created Successfully")