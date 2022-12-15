from flask import Flask, render_template, url_for, request
#from backend.scripts.nn_training_2 import Bi_LSTM
from backend.scripts.backend import Backend

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")


@app.route('/result', methods=['POST', 'GET'])
def result():
    output = request.form.to_dict()

    sentence = output["sentence"]
    algorithm = output["algorithm"]

    if algorithm == "LSTM":
        val = backend.analyze_sentiment(sentence, "LSTM")
    if algorithm == "C-LSTM":
        val = backend.analyze_sentiment(sentence, "C-LSTM")
    elif algorithm == "Bi-LSTM":
        val = backend.analyze_sentiment(sentence, "Bi-LSTM")
    elif algorithm == "SVM":
        val = backend.analyze_sentiment(sentence, "SVM")
    elif algorithm == "Logistic Regression":
        val = backend.analyze_sentiment(sentence, "lr")
    roundedVal = round(val, 3)
    category = "negative"
    if roundedVal >= .66:
        category = "positive"
    elif roundedVal > .33:
        category = "neutral"
    decimalVal = str(roundedVal)

    return render_template('index.html', sentence=sentence, algorithm=algorithm, decimalVal=decimalVal, category=category)


if __name__ == "__main__":
    backend = Backend()
    app.run(debug=True)
