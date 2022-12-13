import os
import pickle
import numpy as np
from gensim.models import Word2Vec
from .preprocess import Preprocess
from .models import padding, compute_w2v_matrix, seq_to_vector, \
                    lstm, c_lstm, bi_lstm, svm, lr


class Backend():
    def __init__(self):
        # Constants
        parent_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        tokenizer_path = parent_dir + '/content/model/Tokenizer.pickle'
        w2v_path = parent_dir + '/content/model/w2v-twitter-100'
        self.padding_length = 60

        # Text preprocessing, tokenizing and w2v embedding
        self.preprocess = Preprocess()
        self.tokenizer = pickle.load(open(tokenizer_path, 'rb'))
        self.w2v_matrix = compute_w2v_matrix(
            Word2Vec.load(w2v_path), self.tokenizer)

        # Models
        self.lr = lr()
        self.svm = svm()
        self.lstm = lstm()
        self.bi_lstm = bi_lstm()
        self.c_lstm = c_lstm()

    def analyze_sentiment(self, text, selected_model="C-LSTM"):
        # Pre-process text
        processed_text = self.preprocess.preprocess(text)
        # Tokenize text
        tokenized_sequence = self.tokenizer.texts_to_sequences(
            [processed_text])

        # Pure LSTM model
        if selected_model == "LSTM":
            input_seq = padding(tokenized_sequence, self.padding_length)
            result = float(self.lstm(input_seq)[0])

        # C-LSTM model
        elif selected_model == "C-LSTM":
            input_seq = padding(tokenized_sequence, self.padding_length)
            result = float(self.c_lstm(input_seq)[0])

        # Bi-LSTM model
        elif selected_model == "Bi-LSTM":
            input_seq = padding(tokenized_sequence, self.padding_length)
            result = float(self.bi_lstm(input_seq)[0])

        # SVM model
        elif selected_model == "SVM":
            input_seq = padding(tokenized_sequence, self.padding_length)
            vec = seq_to_vector(self.w2v_matrix, np.array([input_seq]))
            result = float(self.svm.predict(vec)[0])
        
        # Logistic Regression model
        elif selected_model == "lr":
            input_seq = padding(tokenized_sequence, self.padding_length)
            vec = seq_to_vector(self.w2v_matrix, np.array([input_seq]))
            result = float(self.lr.predict(vec)[0])

        else:
            print("No such model")
            result = -1

        # Return score (0 - 1)
        return result


if __name__ == "__main__":
    backend = Backend()
    s1 = backend.analyze_sentiment("The game was wonderfullll!", "LSTM")
    print("With LSTM, the score for sentence 1 is", s1)
    s1 = backend.analyze_sentiment("The game was wonderfullll!", "C-LSTM")
    print("With C-LSTM, the score for sentence 1 is", s1)
    s2 = backend.analyze_sentiment(
        "I really don't like the music today :(", "Bi-LSTM")
    print("With Bi-LSTM, the score for sentence 2 is", s2)
    s2 = backend.analyze_sentiment(
        "I really don't like the music today :(", "SVM")
    print("With SVM, the score for sentence 2 is", s2)
