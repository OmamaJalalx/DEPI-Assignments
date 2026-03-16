from fastapi import FastAPI
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = FastAPI()

model = joblib.load(r"D:\DEPI\Technical\nlp project\model.pkl")
vectorizer = joblib.load(r"D:\DEPI\Technical\nlp project\vectorizer.pkl")
#preprocess = joblib.load(r"D:\DEPI\Technical\nlp project\preprocess.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def process_text(text):

    text = text.lower()  #case folding

    text = re.sub(r'[^a-zA-Z\s]', '', text) #remove any non alphabetical character

    words = word_tokenize(text) #tokenization

    words = [word for word in words if word not in stop_words] #stop word removal

    words = [lemmatizer.lemmatize(word) for word in words] #lemmatization

    return " ".join(words)


@app.post("/predict")
def predict(data: dict):

    text = data["text"]

    cleaned = process_text(text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    return {"prediction": prediction}