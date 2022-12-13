# Data loading
from data import load_twitter_data

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB

# Tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

# Utils
import numpy as np
import pickle
import itertools

# Plots
import matplotlib.pyplot as plt
import seaborn as sn


# Load data
X_data, y_data = load_twitter_data()
'''
# Use only a portion
percent = 0.05
np.random.seed(0)
idx = np.random.choice(X_data.shape[0], int(percent * X_data.shape[0]), replace=False)
X_data = X_data[idx]
y_data = y_data[idx]
'''

# Split data
text_X_train, text_X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.05, random_state=0)
print("Training set size:", len(y_train))
print("Testing set size:", len(y_test))


# Vectorizing
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(text_X_train)
print(f'Vectoriser fitted.')

X_train = vectoriser.transform(text_X_train)
X_test  = vectoriser.transform(text_X_test)
print(f'Data Vectorized.')


'''
from gensim.models import Word2Vec
w2v_model_path = 'models/w2v-twitter-100'
w2v_model = Word2Vec.load(w2v_model_path)
w2v_size = w2v_model.vector_size

tokenizer_path = 'models/Tokenizer.pickle'
tokenizer = pickle.load(open(tokenizer_path, 'rb'))

from keras_preprocessing.sequence import pad_sequences
input_length = 60
X_train = pad_sequences(tokenizer.texts_to_sequences(text_X_train), maxlen=input_length)
X_test  = pad_sequences(tokenizer.texts_to_sequences(text_X_test) , maxlen=input_length)

from tqdm import tqdm
from scipy import sparse
def compute_w2v_matrix(w2v_model, tokenizer):
    input_dim = tokenizer.num_words
    output_dim = w2v_model.vector_size
    # Create embedding matrix
    embedding_matrix = np.zeros((input_dim, output_dim))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    
    # Convert to sparse embedding matrix
    # embedding_matrix = sparse.csr_matrix(embedding_matrix)
    return embedding_matrix

def seq_to_vector(w2v_matrix, X, batch_size=20000):
    num_batch = int(np.ceil(X.shape[0] / batch_size))
    num_feature = w2v_matrix[X[0]].flatten().shape[0]
    batch_Xs = []

    # Convert it batch by batch to avoid exceeding memory
    for b in tqdm(range(num_batch)):

        # get current batch
        batch_X = X[b*batch_size : (b+1)*batch_size]
        # init feature matrix
        transformed_X = np.zeros((batch_X.shape[0], num_feature))

        # vectorize
        for i, seq in enumerate(batch_X):
            transformed_v = w2v_matrix[seq].flatten()
            transformed_X[i] = transformed_v

        # append to final output
        batch_Xs.append(sparse.csr_matrix(transformed_X))
    
    vectorized_X = sparse.vstack(batch_Xs)
    return vectorized_X

# Compute word2vec embedding matrix
w2v_matrix = compute_w2v_matrix(w2v_model, tokenizer)
# Transform dataset
X_train = seq_to_vector(w2v_matrix, X_train)
X_test = seq_to_vector(w2v_matrix, X_test)
'''

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


# Training Models

BNBmodel = BernoulliNB(alpha = 2)
BNBmodel.fit(X_train, y_train)
y_pred = BNBmodel.predict(X_test)
y_pred = np.where(y_pred>=0.5, 1, 0)
plot_confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
plt.savefig("evaluation_BNB.png")


SVCmodel = LinearSVC()
SVCmodel.fit(X_train, y_train)
y_pred = SVCmodel.predict(X_test)
y_pred = np.where(y_pred>=0.5, 1, 0)
plot_confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
plt.savefig("evaluation_SVC.png")


# Save models
model_path ='models/BernoulliNB.sav'
pickle.dump(BNBmodel, open(model_path, 'wb'))

# Save models
model_path ='models/SVM.sav'
pickle.dump(SVCmodel, open(model_path, 'wb'))
