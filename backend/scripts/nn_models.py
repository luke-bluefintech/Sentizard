import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Embedding, Conv1D, Concatenate, \
                         Bidirectional, LSTM, GlobalMaxPool1D


# Embedding layer
def EmbeddingLayer(tokenizer, w2v_model, input_length):
    vocab_size = tokenizer.num_words
    w2v_size = w2v_model.vector_size
    
    # Create embedding matrix
    embedding_matrix = np.zeros((vocab_size, w2v_size))
    # print("Embedding Matrix Shape:", embedding_matrix.shape)

    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]

    # Return embedding layer
    return Embedding(input_dim = vocab_size,
                     output_dim = w2v_size,
                     weights = [embedding_matrix],
                     input_length = input_length,
                     trainable = False)


# LSTM Model
def Pure_LSTM(tokenizer, w2v_model, input_length):
    model = Sequential()
    model.add(EmbeddingLayer(tokenizer, w2v_model, input_length))
    model.add(Dropout(0.5))
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    
    return model


# Bi-LSTM + CNN
def Bi_LSTM(tokenizer, w2v_model, input_length):
    model = Sequential()
    model.add(EmbeddingLayer(tokenizer, w2v_model, input_length))

    model.add(Bidirectional(LSTM(256, dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.2, return_sequences=True)))
    
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(GlobalMaxPool1D())

    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    return model


# C-LSTM Model
def C_LSTM(tokenizer, w2v_model, input_length):
    model = Sequential()
    model.add(EmbeddingLayer(tokenizer, w2v_model, input_length))
    model.add(Dropout(0.2))

    conv3 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(model.output)
    conv5 = Conv1D(64, kernel_size=5, padding='same', activation='relu')(model.output)
    conv7 = Conv1D(64, kernel_size=7, padding='same', activation='relu')(model.output)
    concatted = Concatenate(axis=-1)([conv3, conv5, conv7])

    out = LSTM(256, activation='relu', recurrent_dropout=0.4)(concatted)
    out = Dropout(0.2)(out)

    out = Dense(64, activation='relu')(out)
    out = Dense(1, activation='sigmoid')(out)
    
    model = Model(inputs=model.input, outputs=out)
    return model
