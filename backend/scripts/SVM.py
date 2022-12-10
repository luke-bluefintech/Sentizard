# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:31:11 2022

@author: oz_ge
"""
import pickle
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
from nn_models import padding
import os
####EVALUATION####
'''
with open('svm_linear.pickle', 'rb') as handle:
    svm_linear = pickle.load(handle)
    
with open('X_test.pickle', 'rb') as handle:
    X_test = pickle.load(handle)    
    
with open('y_test.pickle', 'rb') as handle:
    y_test = pickle.load(handle)     

   
'''
#preds = svm_linear.predict(X_test)
#report = classification_report(y_train, preds, output_dict=True)
#confusion_matrix(y_test, preds)




text = 'The game was wonderfullll!'
def linear_svm(text,tokenizer):
    svm_path = os.getcwd()+ "/Content/svm_linear.pickle"
    
    with open(svm_path, 'rb') as handle:
        svm_linear = pickle.load(handle)
    
    tokenized = pad_sequences(tokenizer.texts_to_sequences([text]))
    input_matrix = padding(tokenized)
    pred = svm_linear.predict(input_matrix)[0]
    #pred_final = 1 -  sum(pred)/len(pred)
    return pred






