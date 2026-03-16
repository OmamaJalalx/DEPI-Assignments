# NLP Spam Classifier

This project builds a traditional NLP pipeline for SMS spam detection.

## Features
- Text preprocessing
- Bag of Words / TF-IDF
- Logistic Regression, Naive Bayes, SVM
- FastAPI deployment

## API

Run the API:

uvicorn app:app --reload

Then open:

http://127.0.0.1:8000/docs

## Example Request

{
"text": "Congratulations! You won a free ticket"
}

Response:

{
"prediction": "spam"
}
