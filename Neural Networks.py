import numpy as np # linear algebra
import pandas as pd 
import os
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('/dataset/Training.csv')
test= pd.read_csv('/dataset/Testing.csv')
train.head()

train.drop('Unnamed: 133', axis=1, inplace=True)


X_train= train.drop('prognosis', axis=1)
X_test= test.drop('prognosis', axis=1)

y_train= np.array(train['prognosis'])
y_test= np.array(test['prognosis'])

y_train_enc= pd.get_dummies(y_train)

y_test_enc= pd.get_dummies(y_test)

model= Sequential()
model.add(Dense(64, activation='relu', input_shape= (X_train.shape[1], )))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(y_train_enc.shape[1], activation='softmax'))

model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping_monitor= EarlyStopping(patience=2, monitor='val_accuracy')
model.fit(X_train, y_train_enc, batch_size=120, epochs=30, validation_split=0.3, callbacks=[early_stopping_monitor])

model.evaluate(X_test, y_test_enc, batch_size=1, steps=5)

prediction= model.predict_classes(X_test)

Xnew= train.drop('prognosis', axis=1)
ynew= train['prognosis']

def create_model(learning_rate, activation):
    model2= Sequential()
    my_opt= Adam(lr= learning_rate)
    model2.add(Dense(64, activation=activation, input_shape=(X_train.shape[1],)))
    
    model2.add(Dense(y_train_enc.shape[1], activation='softmax'))
    model2.compile(optimizer= my_opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model2

modelnew= KerasClassifier(build_fn= create_model, epochs=30, batch_size=100, validation_split=0.3)

params = {'activation': ['relu', 'tanh'], 'batch_size': [32, 128, 256], 
          'epochs': [10], 'learning_rate': [0.1, 0.01, 0.001]}


random_search = RandomizedSearchCV(modelnew, param_distributions = params, cv = 5)
random_search.fit(Xnew, ynew)

Xtestnew= test.drop('prognosis', axis=1)
ytestnew= test['prognosis']

random_search.best_estimator_.score(Xtestnew, ytestnew)
pred= random_search.best_estimator_.predict(X_test)
