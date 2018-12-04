from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, GRU


class Stacked_GRU_Regression:
    @staticmethod
    def build(embedding_size):
        model = Sequential()
        model.add(GRU(units=embedding_size))
        model.add(GRU(units=embedding_size))
        model.add(GRU(units=embedding_size))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))

        return model;