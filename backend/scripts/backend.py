import pickle
from preprocess import Preprocess
from nn_models import padding, c_lstm
from SVM import linear_svm
import os


class Backend():
    def __init__(self):
        tokenizer_path = os.getcwd()+ "/Tokenizer.pickle"

        # Text preprocessing and tokenizing
        self.preprocess = Preprocess()
        self.tokenizer = pickle.load(open(tokenizer_path, 'rb'))

        # Models
        self.c_lstm = c_lstm()



    def analyze_sentiment(self, text, selected_model="C-LSTM"):
        # Pre-process text
        processed_text = self.preprocess.preprocess(text)
        # Tokenize text
        tokenized_sequence = self.tokenizer.texts_to_sequences([processed_text])
        #tokenized_sequence = tokenizer.texts_to_sequences([processed_text])
        # C-LSTM model
        if selected_model == "C-LSTM":
            model = c_lstm()
            input_matrix = padding(tokenized_sequence)
            result = float( model([input_matrix])[0] )
        elif selected_model == 'SVM_linear':
            result = linear_svm(text, self.tokenizer)
            if result == 1:
                result = 'Positive'
            else:
                result='Negative'


        # Return score (0 - 1)
        return result


if __name__ == "__main__":
    backend = Backend()
    s1 = backend.analyze_sentiment("The game was wonderfullll!", "C-LSTM")
    print("The score for sentence 1 is", s1)
    s2 = backend.analyze_sentiment("I really don't like the music today :(", "C-LSTM")
    print("The score for sentence 2 is", s2)
    s3 = backend.analyze_sentiment("The game was wonderfullll!", "SVM_linear")
    print("The score for sentence 1 is (with linear svm)", s3)
    s4 = backend.analyze_sentiment("I had a terrible day today, not feeling great...", "SVM_linear")
    print("The score for sentence 3 is (with linear svm)", s4)

