from keras.layers import Dense
from keras.models import Sequential


def get_nn():
    model = Sequential()
    model.add(Dense(units=200,input_dim=80*80, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model