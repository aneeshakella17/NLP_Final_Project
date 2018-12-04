from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, GRU
from keras.layers.embeddings import Embedding
import math

class LSTM_Regression:
    @staticmethod
    def build(vocab_size, embedding_size, the_weights):
        model = Sequential()
        model.add(Embedding(output_dim = embedding_size, input_dim = vocab_size, weights = [the_weights]))
        model.add(LSTM(units=embedding_size))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))

        return model;