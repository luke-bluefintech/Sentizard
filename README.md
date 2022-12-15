# Sentizard 

Tweet sentiment analysis with machine learning and deep learning methods

### Prerequisite

- Install scikit-learn, Tensorflow and Keras

- Git clone this project

- Download dataset from [Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/datasets/kazanova/sentiment140), and put the csv file under <u>YOUR PATH\Twitter-Sentiment-Analysis\Backend\content\dataset\



## Training



The model used to analyze twitter sentiment are BernoulliNB, SVM, LSTM, C-LSTM and BiLSTM-CNN.



To train the models, please run with `python nn_training.py` or `python ml_training.py`.

## Results
| Vectorizing Method | Model | Accuracy | 
| ------------------ | ----- | -------- | 
| TF-IDF | Bernoulli Naive Bayes | 0.80 | 
| TF-IDF | SVM | 0.82 | 
| Word2Vec | Bernoulli Naive Bayes | 0.54 | 
| Word2Vec | SVM | 0.69 | 
| Word2Vec | LSTM | 0.828 | 
| Word2Vec | C-LSTM | 0.836 | 
| Word2Vec | Bi-LSTM | 0.846 |

## Using the website

Please make sure that you have first installed flask. Then, from the sentizard directory folder, please run `python app.py` and wait 30 seconds for the local server to start. Then, type http://localhost:5000/ into your browser, and the website should display with full funcitonality

