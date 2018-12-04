from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, GRU
from keras.layers.embeddings import Embedding
import math

class Stacked_GRU_Classification:
    @staticmethod
    def build(embedding_size):
        model = Sequential()
        model.add(GRU(units=embedding_size))
        model.add(GRU(units=embedding_size))
        model.add(GRU(units=embedding_size))
        model.add(Dropout(0.2))
        model.add(Dense(5))
        model.add(Activation("softmax"))

        return model;