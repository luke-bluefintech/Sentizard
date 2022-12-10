from flask import Flask, render_template, url_for, request
# load backend code
import os
import sys
from backend.scripts.backend import Backend


# get current path
curr_dir = os.path.abspath('')
app = Flask(__name__, template_folder=curr_dir)


@app.route('/')
@app.route('/home')
def home():
    return render_template("index_proto.html")


@app.route('/result', methods=['POST', 'GET'])
def result():
    # Get the submitted sentence
    input_data = request.form.to_dict()
    # print(output)
    sentence = input_data["sentence"]
    model_name = "C-LSTM"

    ##############################
    # Analyze the sentiment of the sentence
    val = backend.analyze_sentiment(sentence, model_name)
    val_str = str(val)
    ##############################

    return render_template('index_proto.html', res=val_str)


if __name__ == "__main__":
    # Load backend
    backend = Backend()

    # Launch website
    app.run(debug=True)
