# import essential libraries
import pandas as pd
import re
from joblib import dump
from nltk import download
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# download nltk data for stopwords
download("stopwords")

# load the dataset
df = pd.read_csv("data/Spam SMS Collection", sep="\t", names=["label", "message"])

corpus = []  # empty list to store corpus of messages
ps = PorterStemmer()  # word stemmer based on the Porter stemming algorithm

for idx in range(df.shape[0]):
    message = re.sub(
        pattern="[^a-zA-Z]", repl=" ", string=df.message[idx]
    )  # clean special character from the message
    message = message.lower()  # convert the entire message into lower case
    words = message.split()  # tokenize the review by words
    words = [
        word for word in words if word not in set(stopwords.words("english"))
    ]  # Remove stop words
    words = [ps.stem(word) for word in words]  # stemming the words
    message = " ".join(words)  # join the stemmed words
    corpus.append(message)  # append a corpus of messages

# convert a collection of text documents to a matrix of token counts
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

# extracting dependent variable from the dataset
y = pd.get_dummies(df["label"])
y = y.iloc[:, 1].values

# dump CountVectorizer as a pickle file
dump(cv, open("models/transform.pkl", "wb"))

# split the dataset into training & testing sets by 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# fit multinomial naive bayes to the training set
classifier = MultinomialNB(alpha=0.3)
classifier.fit(X_train, y_train)

# dump Multinomial Naive Bayes model as a pickle file
dump(classifier, open("models/spamClassifier.pkl", "wb"))
