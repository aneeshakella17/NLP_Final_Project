import json
import numpy as np
from numpy import array
from keras.optimizers import SGD
from gensim.models import KeyedVectors
from nn import LSTM_Classification
from sklearn.preprocessing import LabelBinarizer
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing import sequence
import tensorflow
import pandas as pd


tokenizer = RegexpTokenizer('\w+(?:(?:\'|-)\w+)?')
stop_words = [];
opt = SGD(lr=1e-05)

print("Collecting Stop Words")

with open('stopwords.txt','r') as f:
    for line in f:
        for word in line.split():
            stop_words.append(word)

print("Collecting Google News NLP")
google_vecs = KeyedVectors.load_word2vec_format('GoogleNewsNLP.bin', binary=True, limit = 200000)
weights = google_vecs.wv.syn0;

# print(, embedding_size);

def json_readr(file):
    data = [];
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))
    return data;

#reveiw_tokenize(text)
# Input: text - String containing review
# Output: tokens - Tokenized version of text
#                  All punctuation is removed except for dashes and apostrophes in words
#         Stop Words removed
def tokenize(text):
    vect = tokenizer.tokenize(text.lower())
    tokens = [w for w in vect if w not in stop_words]
    return tokens


def w2v_vectorize(tokens):
    vects=[]
    ct = 0
    for w in tokens:
        try:
            vects.append(google_vecs[w])
            ct += 1
        except KeyError:
            continue
    return vects;


print("Collecting our data ...")
data = json_readr('yelp_dataset_small_sample.json')
length_of_data = len(data);

train_data_size = int(length_of_data * 0.75);
max_words = 500;
top_words = 5000;

print("Vectorizing the data ...")
X = [w2v_vectorize(tokenize(review['text'])) for review in data];
Y_train = [review['stars'] for review in data[0:train_data_size]];
Y_test = [review['stars']for review in data[train_data_size::]];

X = sequence.pad_sequences(X);
X_train = X[0:train_data_size]
X_test = X[train_data_size::]

le = LabelBinarizer();
Y_train, Y_test = le.fit_transform(Y_train), le.fit_transform(Y_test);

lstm_model = LSTM_Classification.build(300);
lstm_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print("Fitting the data")
print("NP Shape", np.shape(X_train));
history = lstm_model.fit(np.array(X_train), np.array(Y_train), validation_data=(np.array(X_test), np.array(Y_test)), epochs=100, batch_size = 100, verbose = 1)
val_acc=history.history['val_acc']
val_loss=history.history['val_loss']
table=pd.DataFrame({'val_acc':val_acc, 'val_loss':val_loss})
table.to_csv('file_name.csv', sep=',');
lstm_model.save("lstm_model_weights.hdf5");
