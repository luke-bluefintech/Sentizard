import os
import pickle
import numpy as np
import keras
from keras_preprocessing.sequence import pad_sequences
from scipy import sparse
# import tensorflow as tf
# tf.get_logger().setLevel('WARNING')


def padding(sequence, maxlen=60):
    # Padding sequences
    return pad_sequences(sequence, maxlen=maxlen)

def compute_w2v_matrix(w2v_model, tokenizer):
    input_dim = tokenizer.num_words
    output_dim = w2v_model.vector_size
    # Create embedding matrix
    embedding_matrix = np.zeros((input_dim, output_dim))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    
    return embedding_matrix

def seq_to_vector(w2v_matrix, token_seqs):
    # init feature matrix
    num_feature = w2v_matrix[token_seqs[0]].flatten().shape[0]
    transformed_X = np.zeros((token_seqs.shape[0], num_feature))

    # vectorize
    for i, seq in enumerate(token_seqs):
        transformed_v = w2v_matrix[seq].flatten()
        transformed_X[i] = transformed_v

    transformed_X = sparse.csr_matrix(transformed_X)
    return transformed_X


def c_lstm():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = parent_dir + '/content/model/C-LSTM.h5'
    model = keras.models.load_model(model_path)
    return model


def svm():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = parent_dir + '/content/model/SVM.sav'
    model = pickle.load(open(model_path, 'rb'))
    return model

def lr():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = parent_dir + '/content/model/lr.sav'
    model = pickle.load(open(model_path, 'rb'))
    return model
