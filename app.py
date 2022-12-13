from flask import Flask, render_template, url_for, request
#from backend.scripts.nn_training import Bi_LSTM
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
    date1 = output["date1"]
    date2 = output["date2"]
    city = output["city"]
    algorithm = output["algorithm"]

    if algorithm == "C-LSTM":
        val = backend.analyze_sentiment(sentence, "C-LSTM")
    elif algorithm == "SVM":
        val = backend.analyze_sentiment(sentence, "SVM")
    elif algorithm == "BiLSTM":
        print("BiLSTM!")
        #val = Bi_LSTM()
    sentence = str(val)

    return render_template('index.html', sentence=sentence, date1=date1, date2=date2, city=city, algorithm=algorithm)


if __name__ == "__main__":
    backend = Backend()
    app.run(debug=True)
