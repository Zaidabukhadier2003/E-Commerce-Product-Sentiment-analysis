import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('ecommerce_reviews.csv')

# Display the first few rows of the DataFrame
print("Sample Data:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize the distribution of sentiments
sentiment_counts = df['Sentiment'].value_counts()
sentiment_counts.plot(kind='bar', title='Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Text Preprocessing
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['ReviewText'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, df['Sentiment'], test_size=0.2, random_state=42)

# Naive Bayes Classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)

# Word Cloud for positive and negative sentiments
positive_reviews = ' '.join(df[df['Sentiment'] == 'positive']['ReviewText'])
negative_reviews = ' '.join(df[df['Sentiment'] == 'negative']['ReviewText'])

wordcloud_positive = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = 'english', 
                min_font_size = 10).generate(positive_reviews)
wordcloud_negative = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = 'english', 
                min_font_size = 10).generate(negative_reviews)

# Display Word Clouds
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud_positive) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title('Word Cloud - Positive Sentiments')
plt.show()

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud_negative) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title('Word Cloud - Negative Sentiments')
plt.show()
