#!/usr/bin/env python
# coding: utf-8

# ### Install required packages for training model
# 

# In[11]:


get_ipython().system('pip install gensim --upgrade')
get_ipython().system('pip install tensorflow-gpu --upgrade')
get_ipython().system('pip install keras --upgrade')
get_ipython().system('pip install pandas --upgrade')


# ### Import packages

# In[12]:


# DataFrame
import pandas as pd

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Word2vec
from gensim.models import Word2Vec

# Keras
import keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Input, Activation, Dense, Dropout, Embedding, Conv1D, Concatenate, Bidirectional, LSTM, GlobalMaxPool1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Utils
import os
import re
import pickle
import numpy as np
import time
import itertools

# Plots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


# Test if GPU is avaliable
import tensorflow as tf
tf.config.list_physical_devices()


# # Data Preparation

# ### Import Data

# The dataset file location should be
# 
# "../content/dataset/training.1600000.processed.noemoticon.csv"

# In[3]:


# Read raw dataset
dataset_columns = ["sentiment", "ids", "date", "flag", "user", "text"]
dataset_encoding = "ISO-8859-1"
dataset = pd.read_csv('../content/dataset/training.1600000.processed.noemoticon.csv',
                      encoding=dataset_encoding , names=dataset_columns)
dataset.head(5)


# In[4]:


# Removing the unnecessary columns.
dataset = dataset[['sentiment','text']]

# Replace value 4 with 1 (positive)
dataset['sentiment'] = dataset['sentiment'].replace(4,1)


# In[5]:


ax = dataset.groupby('sentiment').count().plot(kind='bar', title='Distribution of data', legend=False)
ax = ax.set_xticklabels(['Negative', 'Positive'], rotation=0)


# ### Pre-Process dataset

# In[6]:


# Load Enlgish contraction dictionary
contractions = pd.read_csv('../content/dataset/contractions.csv', index_col='Contraction')
contractions.index = contractions.index.str.lower()
contractions.Meaning = contractions.Meaning.str.lower()
contractions_dict = contractions.to_dict()['Meaning']

# Define text cleaning pattern
urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
userPattern       = '@[^\s]+'
hashtagPattern    = '#[^\s]+'
alphaPattern      = "[^a-z0-9<>]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

# Define emojis cleaning pattern
smileemoji        = r"[8:=;]['`\-]?[)d]+"
sademoji          = r"[8:=;]['`\-]?\(+"
neutralemoji      = r"[8:=;]['`\-]?[\/|l*]"
lolemoji          = r"[8:=;]['`\-]?p+"


# In[7]:


def preprocess(text):
    
    # 1, Convert to lower case
    text = text.lower()

    # 2, Replace all URls with '<url>'
    text = re.sub(urlPattern, '<url>', text)
    
    # 3, Replace all @USERNAME to '<user>'.
    text = re.sub(userPattern, '<user>', text)
    
    # 4, Replace 3 or more consecutive letters by 2 letter.
    text = re.sub(sequencePattern, seqReplacePattern, text)

    # 5, Replace all emojis.
    text = re.sub(r'<3', '<heart>', text)
    text = re.sub(smileemoji, '<smile>', text)
    text = re.sub(sademoji, '<sadface>', text)
    text = re.sub(neutralemoji, '<neutralface>', text)
    text = re.sub(lolemoji, '<lolface>', text)

    # 6, Remove Contractions
    for contraction, replacement in contractions_dict.items():
        text = text.replace(contraction, replacement)

    # 7, Removing Non-Alphabets and replace them with a space
    text = re.sub(alphaPattern, ' ', text)

    # 8, Adding space on either side of '/' to seperate words.
    text = re.sub(r'/', ' / ', text)
    
    return text


# In[8]:


get_ipython().run_cell_magic('time', '', "\n# Clean up the text and store it in a new field\ndataset['processed_text'] = dataset.text.apply(preprocess)\n")


# In[168]:


# Show a sample processed result
sample_idx = 10
print("Original  Text: ", dataset.iloc[sample_idx][1])
print("Processed Text: ", dataset.iloc[sample_idx][2])


# ### Split training and testing set

# In[169]:


X_data, y_data = np.array(dataset['processed_text']), np.array(dataset['sentiment'])

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                    test_size = 0.05, random_state = 0)
print("TRAIN size:", len(y_train))
print("TEST size:", len(y_test))


# ### Word embedding

# In[170]:


get_ipython().run_cell_magic('time', '', "\nw2v_model_path = '../content/model/w2v-twitter-100'\n\n# Load the word2vec model if previously trained\nif os.path.exists(tokenizer_path):\n    w2v_model = Word2Vec.load(w2v_model_path)\n\n# Train a new word2vec model\nelse:\n    # Parameters\n    w2v_size = 100\n    w2v_window = 7\n    w2v_min_count = 10\n\n    # Training set for w2v model\n    w2v_training_data = list(map(lambda x: x.split(), X_train))\n\n    # Defining w2v model and training it.\n    w2v_model = Word2Vec(w2v_training_data,\n                         vector_size = w2v_size,\n                         workers = w2v_window,\n                         min_count = w2v_min_count)\n")


# In[171]:


# Demonstrate result
print("Vocabulary Length:", len(w2v_model.wv.key_to_index))
print("Similar words to 'love'")
w2v_model.wv.most_similar("love")


# ### Tokenize text

# In[9]:


get_ipython().run_cell_magic('time', '', '\ntokenizer_path = \'../content/model/Tokenizer.pickle\'\n\n# Load the tokenizer if previously trained\nif os.path.exists(tokenizer_path):\n    tokenizer = pickle.load(open(tokenizer_path, \'rb\'))\n    \n# Train a tokenizer\nelse:\n    vocab_size = 100000\n    tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")\n    tokenizer.fit_on_texts(X_data)\n    tokenizer.num_words = vocab_size\n\n# vocab_size = len(tokenizer.word_index) + 1\nprint("Total words", tokenizer.num_words)\n')


# ### Padding text

# In[10]:


input_length = 60

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=input_length)
X_test  = pad_sequences(tokenizer.texts_to_sequences(X_test) , maxlen=input_length)

print("X_train.shape:", X_train.shape)
print("X_test.shape :", X_test.shape)


# # Model Building And Training

# ### Build Model

# In[ ]:


# Embedding layer
def EmbeddingLayer():
    
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


# In[ ]:


# Bi-LSTM + CNN
def Bi_LSTM():
    model = Sequential([
        EmbeddingLayer(),
        Bidirectional(LSTM(60, dropout=0.3, return_sequences=True)),
        Bidirectional(LSTM(60, dropout=0.3, return_sequences=True)),
        Conv1D(60, kernel_size=5, activation='relu'),
        GlobalMaxPool1D(), 
        Dense(1, activation='sigmoid'),
    ])
    
    return model

# C-LSTM Model
def C_LSTM():
    input_layer = Input(shape = (input_length))
    embedding_layer = EmbeddingLayer()
    x = embedding_layer(input_layer)
    x = Dropout(0.2)(x)

    conv2 = Conv1D(16, kernel_size=2, activation='relu')(x)
    conv3 = Conv1D(16, kernel_size=3, activation='relu')(x)
    conv5 = Conv1D(16, kernel_size=5, activation='relu')(x)
    conv7 = Conv1D(16, kernel_size=7, activation='relu')(x)
    conv_len = conv7.shape[1]
    x = Concatenate(axis=-1)([conv2[:,:conv_len,:], 
                              conv3[:,:conv_len,:], 
                              conv5[:,:conv_len,:], 
                              conv7[:,:conv_len,:]])

    x = LSTM(128, dropout=0.2, activation='relu', return_sequences=True)(x)
    x = LSTM(128, dropout=0.2, activation='relu')(x)

    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=x)
    
    return model

# LSTM Model
def Pure_LSTM():
    model = Sequential()
    model.add(EmbeddingLayer())
    model.add(Dropout(0.5))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    return model


# ### Loading or Training Model

# In[ ]:


load_model = True
# Load a model if already trained
if load_model:
    model = keras.models.load_model(model_path)


# Train a new model
else:
    model = C_LSTM()
    # model = Bi_LSTM()
    # model = Pure_LSTM()
    
    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    
    
model.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nif not load_model:\n\n    # Training parameters\n    epochs = 6\n    batch_size = 512\n\n    # Define callbacks\n    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),\\\n                 EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]\n\n    # Start training\n    history = model.fit(X_train, y_train,\n                        batch_size = batch_size,\n                        epochs = epochs,\n                        validation_split = 0.1,\n                        verbose = 1,\n                        callbacks = callbacks)\n")


# ### Evaluate

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nscore = model.evaluate(X_test, y_test, batch_size = batch_size)\nprint()\nprint("ACCURACY:",score[1])\nprint("LOSS:",score[0])\n')


# In[179]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()


# ### Predict samples

# In[180]:


def predict(text):
    # Pre-process text
    processed_text = preprocess(text)
    # Tokenize text
    input_matrix = pad_sequences(tokenizer.texts_to_sequences([processed_text]), maxlen=input_length)
    # Predict
    score = model.predict([input_matrix])[0]
    
    return float(score)


# In[181]:


predict("I love the music!!")


# In[182]:


predict("I hate the rain :(")


# 

# In[183]:


predict("i don't know what i'm doing")


# # Model Analysis

# ### Confusion Matrix

# In[184]:


def plot_confusion_matrix(y_test, y_pred):
    # Compute and plot the Confusion matrix
    print(y_test)
    print(y_pred)
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


# In[185]:


get_ipython().run_cell_magic('time', '', '\n# Predicting on the Test dataset\ny_pred = model.predict(X_test, batch_size = batch_size)\n\n# Converting prediction to reflect the sentiment predicted.\ny_pred = np.where(y_pred>=0.5, 1, 0)\n')


# In[186]:


# Printing out the Evaluation metrics. 
plot_confusion_matrix(y_test, y_pred)


# ### Classification Report

# In[187]:


print(classification_report(y_test, y_pred))


# ### Save model

# In[188]:


# Saving Word2Vec-Model
w2v_model.save(w2v_model_path)

# Saving the tokenizer
with open(tokenizer_path, 'wb') as file:
    pickle.dump(tokenizer, file)

# Saving the TF-Model.
model.save(model_path)

