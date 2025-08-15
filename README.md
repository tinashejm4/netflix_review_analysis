This project uses machine learning classifiers to analyze and predict the sentiment of Netflix reviews from the Google Play Store. 
By leveraging text preprocessing, TF-IDF vectorization, and classification algorithms like Logistic Regression, the model categorizes 
user reviews into positive, neutral, or negative sentiments. This work matters because it allows companies to automatically process 
large volumes of customer feedback, enabling faster decision-making, improved product quality, and proactive customer support.

# Setup Instructions
git clone https://github.com/tinashejm4/netflix-sentiment-analysis.git

cd netflix-sentiment-analysis

#loading the model and vectorizer

# Load your saved model 
import joblib
model = joblib.load("logistic_regression_model.pkl")

vectorizer = joblib.load("tfidf_vectorizer.pkl")


# Test a custom review
review = ["The shows are great but the app keeps crashing."]

review_vectorized = vectorizer.transform(review)

prediction = model.predict(review_vectorized)

print(prediction)

# Future Work
Implement sarcasm detection to improve classification accuracy.
Add support for multilingual reviews.
Deploy the model as a REST API for real-time sentiment prediction.
Incorporate deep learning models like BERT for higher accuracy.
