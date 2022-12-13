# Data loading
from data import load_twitter_data

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Word2vec
from gensim.models import Word2Vec

# Keras
import keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from nn_models import Pure_LSTM, Bi_LSTM, C_LSTM

# Utils
import os
import pickle
import numpy as np
import itertools

# Plots
import matplotlib.pyplot as plt
import seaborn as sn


# Load data
X_data, y_data = load_twitter_data()
text_X_train, text_X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.05, random_state=0)
print("Training set size:", len(y_train))
print("Testing set size:", len(y_test))


# Tokenization
tokenizer_path = 'models/Tokenizer.pickle'

# Load the tokenizer if previously trained
if os.path.exists(tokenizer_path):
    tokenizer = pickle.load(open(tokenizer_path, 'rb'))
    vocab_size = tokenizer.num_words
    
# Train a tokenizer
else:
    vocab_size = 100000
    tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")
    tokenizer.fit_on_texts(text_X_train)
    tokenizer.num_words = vocab_size

# vocab_size = len(tokenizer.word_index) + 1
print("Total words", tokenizer.num_words)

# Tokenize data
tokenized_X_train = tokenizer.texts_to_sequences(text_X_train)
tokenized_X_test = tokenizer.texts_to_sequences(text_X_test)


# Padding to ensure texts have consistent length
input_length = 60
X_train = pad_sequences(tokenized_X_train, maxlen=input_length)
X_test = pad_sequences(tokenized_X_test, maxlen=input_length)


w2v_model_path = 'models/w2v-twitter-100'

# Load the word2vec model if previously trained
if os.path.exists(w2v_model_path):
    w2v_model = Word2Vec.load(w2v_model_path)
    w2v_size = w2v_model.vector_size

# Train a new word2vec model
else:
    # Parameters
    w2v_size = 100
    w2v_window = 7
    w2v_min_count = 10

    # Training set for w2v model
    w2v_training_data = list(map(lambda x: x.split(), text_X_train))

    # Defining w2v model and training it.
    w2v_model = Word2Vec(w2v_training_data,
                         vector_size = w2v_size,
                         workers = w2v_window,
                         min_count = w2v_min_count)


model_path = 'models/Bi_LSTM.h5'

load_model = False
# Load a model if already trained
if load_model:
    model = keras.models.load_model(model_path)

# Train a new model
else:
    # model = Pure_LSTM(tokenizer, w2v_model, 60)
    # model = C_LSTM(tokenizer, w2v_model, 60)
    model = Bi_LSTM(tokenizer, w2v_model, 60)
    
    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    
model.summary()


if not load_model:

    # Training parameters
    epochs = 10
    batch_size = 1024

    # Define callbacks
    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),\
                 EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]

    # Start training
    history = model.fit(X_train, y_train,
                        batch_size = batch_size,
                        epochs = epochs,
                        validation_split = 0.1,
                        verbose = 1,
                        callbacks = callbacks)


# Saving Word2Vec-Model
w2v_model.save(w2v_model_path)

# Saving the tokenizer
with open(tokenizer_path, 'wb') as file:
    pickle.dump(tokenizer, file)

# Saving the TF-Model.
model.save(model_path)


score = model.evaluate(X_test, y_test, batch_size=batch_size)
print("Model loss: ", score[0])
print("Model accuracy: ", score[1])


# Print history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.subplot(2, 1, 1)
plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.legend()

plt.subplot(2, 1, 2)
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.legend()

plt.savefig("history.png")


# Predicting on the Test dataset
y_pred = model.predict(X_test, batch_size=batch_size)
y_pred = np.where(y_pred>=0.5, 1, 0)

def plot_confusion_matrix(y_test, y_pred):
    # Compute and plot the Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12,12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    
    classes  = ['Negative', 'Positive']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.title('Confusion Matrix', fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)

plot_confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

plt.savefig("evaluation.png")
