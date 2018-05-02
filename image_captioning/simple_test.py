import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

m = Sequential()
m.add(Dense(32, input_dim = 784))

m = keras.models.Model(m.inputs, Dense(10)(m.output))

m.summary()