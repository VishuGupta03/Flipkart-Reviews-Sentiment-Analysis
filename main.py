import warnings 
warnings.filterwarnings('ignore') 

import pandas as pd 
import re 
import seaborn as sns 
from sklearn.feature_extraction.text import TfidfVectorizer 
import matplotlib.pyplot as plt 
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 

import nltk 
nltk.download('stopwords') 
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from tqdm import tqdm

# Load the dataset
data = pd.read_csv('flipkart_data.csv') 
print(data.head())

# Display unique ratings
print("Unique ratings:", pd.unique(data['rating']))

# Plotting the distribution of ratings
sns.countplot(data=data, x='rating', order=data.rating.value_counts().index)
plt.show()

# Create the rating label (1 for positive, 0 for negative)
data['label'] = data['rating'].apply(lambda x: 1 if x >= 5 else 0)

def preprocess_text(text_data): 
    preprocessed_text = [] 

    for sentence in tqdm(text_data): 
        # Removing punctuations 
        sentence = re.sub(r'[^\w\s]', '', sentence) 

        # Converting to lowercase and removing stopwords 
        preprocessed_text.append(' '.join(
            token.lower() 
            for token in nltk.word_tokenize(sentence) 
            if token.lower() not in stopwords.words('english')
        )) 

    return preprocessed_text 

# Preprocess the reviews
preprocessed_review = preprocess_text(data['review'].values) 
data['review'] = preprocessed_review
print(data.head())

# Count of each label (1 for positive, 0 for negative)
print(data["label"].value_counts())

# Generate a word cloud for positive reviews
consolidated = ' '.join(word for word in data['review'][data['label'] == 1].astype(str)) 
wordCloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110) 

plt.figure(figsize=(15, 10)) 
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear') 
plt.axis('off') 
plt.show() 

# Feature extraction using TF-IDF
cv = TfidfVectorizer(max_features=2500) 
X = cv.fit_transform(data['review']).toarray()

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, data['label'], test_size=0.33, stratify=data['label'], random_state=42)

# Training the Decision Tree model
model = DecisionTreeClassifier(random_state=0) 
model.fit(X_train, y_train) 

# Testing the model on training data
pred_train = model.predict(X_train) 
train_accuracy = accuracy_score(y_train, pred_train)
print(f"Training Accuracy: {train_accuracy}")

# Plotting the confusion matrix
cm = confusion_matrix(y_train, pred_train) 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True]) 
cm_display.plot() 
plt.show()

# Testing the model on test data
pred_test = model.predict(X_test)
test_accuracy = accuracy_score(y_test, pred_test)
print(f"Test Accuracy: {test_accuracy}")