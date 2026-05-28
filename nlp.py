import pandas as pd
from collections import Counter
from keybert import KeyBERT

df = pd.read_csv("put_your_file_here.csv")
#checking orignal dataset
df.columns = df.columns.str.strip()



df = df.rename(columns={
    "Feedback": "review",
    "Sentiment Label": "label",
    "Ratings": "rating"
})


df = df.drop(columns=["Unnamed: 3"], errors="ignore")




df = df.dropna(subset=["review"])
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")



def rating_to_sentiment(r):
    if r >= 4:
        return "Positive"
    elif r == 3:
        return "Neutral"
    else:
        return "Negative"

df["sentiment_category"] = df["rating"].apply(rating_to_sentiment)



total_reviews = len(df)
avg_rating = round(df["rating"].mean(), 2)

sentiment_distribution = (
    df["sentiment_category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nAverage Rating:", avg_rating)
print("\nSentiment Distribution (%):")
print(sentiment_distribution)



positive_reviews = df[df["sentiment_category"] == "Positive"]



kw_model = KeyBERT()



custom_stopwords = [
    "hospital",
    "good",
    "experience",
    "service",
    "overall",
    "providing"
]



all_keywords = []

for review in positive_reviews["review"]:

    keywords = kw_model.extract_keywords(
        review,
        keyphrase_ngram_range=(2,3),
        stop_words=custom_stopwords,
        top_n=2
    )

    all_keywords.extend([k[0] for k in keywords])


top_k_aspects = Counter(all_keywords).most_common(10)


print("\nTop Positive Aspects:\n")

for phrase, count in top_k_aspects:
    print(f"{phrase} ({count} mentions)")



sample_reviews = positive_reviews["review"].sample(3, random_state=42).tolist()



structured_input = f"""
Hospital Review Analysis

Total Reviews: {total_reviews}

Average Rating: {avg_rating}/5

Sentiment Distribution:
{sentiment_distribution.to_string()}

Top Positive Aspects:
{chr(10).join([f"- {phrase} ({count} mentions)" for phrase, count in top_k_aspects])}

Sample Positive Reviews:
{chr(10).join([f"- {r}" for r in sample_reviews])}
"""


print(structured_input)