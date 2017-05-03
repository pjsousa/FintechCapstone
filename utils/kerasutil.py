import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def ConvBlock(layers, filters, model, add_maxpooling=True):
	for i in range(layers):
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(filters, (3, 3), activation='relu'))

	if add_maxpooling:
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))


def FCBlock(model, block_size=4096,add_batchnorm=False, add_dropout=False):
	if add_batchnorm:
		model.add(BatchNormalization())

	model.add(Dense(block_size, activation='relu'))

	if add_dropout:
		model.add(Dropout(0.3))


