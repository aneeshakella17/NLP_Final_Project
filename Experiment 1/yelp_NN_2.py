
# coding: utf-8

# In[1]:


# Code for performing grid search on neural network parameters
# Haruto Nakai

from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import json
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import Sequence, to_categorical
from keras import backend as K
import tensorflow as tf
from datetime import datetime
import pandas as pd


# In[2]:


google_vecs = KeyedVectors.load_word2vec_format(
    '/media/hnakai/Storage/wordvectors/GoogleNews-vectors-negative300.bin', 
    binary=True, limit=200000)


# In[3]:


tokenizer = RegexpTokenizer('\w+(?:(?:\'|-)\w+)?')
stop_words = set(stopwords.words('english'))
#reveiw_tokenize(text)
# Input: text - String containing review
# Output: tokens - Tokenized version of text
#                  All punctuation is removed except for dashes and apostrophes in words
#                  Stop Words removed
def review_tokenize(text):
    vect = tokenizer.tokenize(text.lower())
    tokens = [w for w in vect if w not in stop_words]
    return tokens

#reveiw_generator()
# Is a generator
# Inputs - start: which line to start, 0(beginning) by default
#          end: which line to end, read until end if -1 (default)
# Yields - For each review in the dataset file:
#            tokens - Tokenized version of text
#            stars - Score given by author
def review_generator(start=0, end=-1):
    f=open('yelp_dataset_small_sample.json')
    count=0
    for line in f:
        if count>=start:
            r_dic = json.loads(line)
            yield (review_tokenize(r_dic["text"]), r_dic["stars"])
        count+=1
        if end!=-1 and count>end:
            break
    
#w2v_vectorize(tokens)
# Input: tokens - Tokenized version of text
# Output: feat_vect - Average of vector representations for each word in text
def w2v_vectorize(tokens):
    feat_vect = numpy.zeros(300)
    ct = 0
    for w in tokens:
        try:
            feat_vect+=google_vecs[w]
            ct+=1
        except KeyError:
            continue
    if ct==0:
        return None
    feat_vect /= ct
    return feat_vect

#vect_rating_gen(start, end, batch_size)
# Inputs - start, end: same as review_generator()
#          batch_size: the number of vector/rating pair in each batch
# Yields - batch_features: a list with vector representations of reviews
#          batch_ratings: a list with star ratings corresponding with ones in batch_features
def vect_rating_gen(start=0, end=-1, batch_size=1000):
    counter=0
    batch_features = numpy.zeros((batch_size, 300))
    #batch_ratings = numpy.zeros((batch_size,1))
    batch_ratings = numpy.zeros((batch_size,5))
    for rev,rat in review_generator(start, end):
        batch_features[counter] = w2v_vectorize(rev)
        #batch_ratings[counter] = rat
        batch_ratings[counter] = to_categorical(rat-1, 5)
        counter+=1
        if counter>=batch_size:
            yield batch_features, batch_ratings
            counter=0
    if counter!=0:
        batch_features=batch_features[0:counter]
        batch_ratings=batch_ratings[0:counter]
        yield batch_features, batch_ratings

#yelpSequence() - wrapper for vect_rating_gen() for parallel processing with Keras
class yelpSequence(Sequence):
    def __init__(self, start=0, end=-1, batch_size=100):
        self.st, self.en = start, end
        if self.en==-1:
            self.en=19000
        self.batch_s = batch_size
        self.gen=vect_rating_gen(start, end, batch_size)
    def __len__(self):
        return numpy.ceil((self.en-self.st)/float(self.batch_s))
    def __getitem__(self,_):
        return self.gen.next()


# In[5]:


#Training

# Parameters
dropout = 0.5
learning_rate = 0.000005
n_hidden = 5
n_epochs = 20

#create_model(n_input)
# Input: n_input - number of inputs the resulting model should handle
# Output: model - neural network model with n_input inputs and 1 output
#                 uses parameters specified above
def create_model(n_input=300, dropout=0.5, learning_rate=0.000005, n_hidden=5):
    model=Sequential()
    n_neurons = int(math.ceil(float(n_input)*2.0/3.0))
    model.add(Dense(n_neurons, activation='relu', input_dim=n_input))
    model.add(Dropout(dropout))
    for _ in range(1, n_hidden):
        model.add(Dense(n_neurons, activation='relu'))
        model.add(Dropout(dropout))
    #model.add(Dense(1, activation='linear'))
    model.add(Dense(5, activation='softmax'))
    #model.compile(loss='mean_squared_error', 
    model.compile(loss='categorical_crossentropy',
        optimizer=RMSprop(lr=learning_rate),
        metrics=['accuracy'])
    return model

#nn_test(dropout, learning_rate, n_hidden, n_epochs)
# Inputs: parameters for the neural network
# Output: acc - list with format [accuracy, loss]
#         hist - a History object that contains validation accuracies/losses after each epoch in training
def nn_test(dropout=0.5, learning_rate=0.000005, n_hidden=5, n_epochs=100):
    training=yelpSequence(0,15200)
    validation=yelpSequence(15200,17100)
    testing=yelpSequence(17100)
    
    tf.reset_default_graph()
    K.clear_session()
    K.set_session(tf.Session())

    model = create_model(300, dropout, learning_rate, n_hidden)
    hist = model.fit_generator(training, epochs=n_epochs, use_multiprocessing=True, workers=16,
                       validation_data=validation, verbose=0)
    acc = model.evaluate_generator(testing)
    return acc, hist


# In[6]:


drs = [0, 0.1, 0.25, 0.5, 0.75, 0.9]
lrs = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
hls = [1,2,3,4,5]
configs = []
for dr in drs:
    for lr in lrs:
        for hl in hls:
            configs.append((dr, lr, hl))

res_acc, res_hist = {}, {}
for dr, lr, hl in configs:
    print (dr,lr,hl)
    st_t = datetime.now()
    acc, hist = nn_test(dr,lr,hl)
    print acc, datetime.now()-st_t
    res_acc[(dr,lr,hl)] = acc
    res_hist[(dr,lr,hl)] = hist


# In[16]:


sort_accs=list(sorted(res_acc.items(), key=lambda x: -x[1][1]))
t_dict = {
    'Parameters':[str(p) for p,_ in sort_accs],
    'Accuracy':[a[1] for _,a in sort_accs],
    'Loss':[a[0] for _,a in sort_accs]
}
df_accs=pd.DataFrame(t_dict, columns=['Parameters', 'Accuracy', 'Loss'])
df_accs.to_csv("csv/yelp_accs.csv",',')


# In[35]:


t_val_acc, t_losses = {}, {}
for p in configs:
    t_val_acc[str(p)]=res_hist[p].history['val_acc']
    t_losses[str(p)]=res_hist[p].history['val_loss']
df_vacc=pd.DataFrame(t_val_acc, columns=[str(p) for p in configs])
df_loss=pd.DataFrame(t_losses, columns=[str(p) for p in configs])
df_vacc.to_csv("csv/yelp_val_accs.csv",',')
df_loss.to_csv("csv/yelp_val_loss.csv",',')

