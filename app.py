import pickle

from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

data = pd.read_csv("protein_data.csv")

data.dropna(inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data["sequence"], data["macromoleculeType"], test_size=0.2, random_state=42)

# Convert the protein sequences to bag-of-words vectors
vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3))
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train a Naive Bayes classifier on the training data
clf = MultinomialNB()
clf.fit(X_train, y_train)

vec_file = 'vectorizer.pickle'
pickle.dump(vectorizer, open(vec_file, 'wb'))

# Save the model
mod_file = 'classification.model'
pickle.dump(clf, open(mod_file, 'wb'))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/classify", methods=["POST"])
def classify():
    sequence = request.form["sequence"]
    loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))

    # load the model
    loaded_model = pickle.load(open('classification.model', 'rb'))

    # make a prediction
    macromoleculeType = loaded_model.predict(loaded_vectorizer.transform([sequence]))

    return render_template("classify.html", sequence=sequence, macromoleculeType=macromoleculeType)

if __name__ == "__main__":
    app.run(debug=True)

