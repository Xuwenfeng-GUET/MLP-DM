# -*- coding: utf-8 -*-
"""
Created on Wed May  5 20:24:49 2021

@author: 666
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, Input
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import time
from tensorflow.tools.docs import doc_controls
from keras import backend
from tensorflow.autograph.experimental import do_not_convert

def a_tanh(x):
    return tf.keras.activations.tanh(-x)

class Constraint:
  def __call__(self, w):
    return w

  def get_config(self):
    return {}

class MinMaxNorm(Constraint):

  def __init__(self, min_value=0.0, max_value=1.0, rate=0.5, axis=0):
    self.min_value = min_value
    self.max_value = max_value
    self.rate = rate
    self.axis = axis

  @doc_controls.do_not_generate_docs
  def __call__(self, w):
    
    norms = backend.sqrt(
        tf.reduce_sum(tf.square(w), axis=self.axis, keepdims=True))
    desired = (
        self.rate * backend.clip(norms, self.min_value, self.max_value) +
        (1 - self.rate) * norms)
    #return backend.clip(w * (desired / (backend.epsilon() + norms)), self.min_value, self.max_value)
    return backend.clip(w* (desired / (backend.epsilon() + norms)), self.min_value, self.max_value) 
  @doc_controls.do_not_generate_docs
  def get_config(self):
    return {
        'min_value': self.min_value,
        'max_value': self.max_value,
        'rate': self.rate,
        'axis': self.axis
    }

"""
B5 = np.array(
    [
    [[1,0,0,0],[-1,1,0,0],[-1,-1,1,0],[-1,-1,-1,-1],[-1,-1,-1,1]],
    [[-1,-1,-1,0],[-1,-1,1,0],[-1,1,0,-1],[-1,1,0,1],[1,0,0,0]],
    [[-1,-1,-1,0],[-1,-1,1,-1],[-1,-1,1,1],[-1,1,0,0],[1,0,0,0]],
    [[-1,-1,0,-1],[-1,-1,0,1],[-1,1,0,0],[1,0,-1,0],[1,0,1,0]],
    [[-1,-1,0,0],[-1,1,0,-1],[-1,1,0,1],[1,0,-1,0],[1,0,1,0]],
    [[-1,0,0,0],[1,-1,0,0],[1,1,-1,0],[1,1,1,-1],[1,1,1,1]],
    [[-1,0,0,0],[1,-1,0,0],[1,1,1,0],[1,1,-1,-1],[1,1,-1,1]],
    [[-1,-1,0,0],[-1,1,0,0],[1,0,1,0],[1,0,-1,-1],[1,0,-1,1]],
    [[-1,-1,0,0],[-1,1,0,0],[1,0,-1,0],[1,0,1,-1],[1,0,1,1]],
    [[-1,0,0,0],[1,-1,-1,0],[1,-1,1,0],[1,1,0,-1],[1,1,0,1]]
    ]
    )
"""
B8 = np.array(
    [
     [-1,-1,0,-1,0,0,0],[-1,-1,0,1,0,0,0],
     [-1,1,0,0,-1,0,0],[-1,1,0,0,1,0,0],
     [1,0,-1,0,0,-1,0],[1,0,-1,0,0,1,0],
     [1,0,1,0,0,0,-1],[1,0,1,0,0,0,1]
     ]
    )

train = pd.read_csv("F:/dataset/UNSW_NB15/UNSW_NB15-master/UNSW_NB15-master/UNSWNB15ARFF 2/UNSWNB15ARFF/UNSWNB15Testing1.csv", header=0)
test = pd.read_csv("F:/dataset/UNSW_NB15/UNSW_NB15-master/UNSW_NB15-master/UNSWNB15ARFF 2/UNSWNB15ARFF/UNSWNB15Training1.csv", header=0)
train_data = train.iloc[:, :-1]
train_target = train.iloc[:, -1]
test_data = test.iloc[:, :-1]
test_target = test.iloc[:, -1]
columns = train_data.columns.tolist()

i = 19
dimension = i
IG = [7,27,12,8,32,28,9,11,1,17,10,6,13,25,24,35,4,16,26,5,18,2,19,15,34,41,14,31,20,3,21,23,22,36,33,40,30,42,39,29,37,38]#information gain
IG = [a-1 for a in IG]
train_data = train_data.iloc[:, IG[:dimension]]
test_data = test_data.iloc[:, IG[:dimension]]
columns1 = train_data.columns.tolist()
#print(train_data.shape)

min_max_scaler = MinMaxScaler()
train_data = min_max_scaler.fit_transform(train_data)
test_data = min_max_scaler.fit_transform(test_data)
train_data = -train_data
test_data = -test_data

train_target = to_categorical(train_target)
test_target = to_categorical(test_target)
#K5 = np.dot(np.linalg.inv(np.diag(np.linalg.norm(B5[j], ord=1, axis=1))),B5[j])
K8 = np.dot(np.linalg.inv(np.diag(np.linalg.norm(B8, ord=1, axis=1))),B8)
batch_size = 32
inputs_dim = train_data.shape[1]
middle_dim_8 = 7
outputs_dim = train_target.shape[1]
start2 = time.time()
inputs = Input(shape = (inputs_dim,))#x
# attention(B,K,h) = v * softmax(K * tanh(S * x - t))
x = Dense(middle_dim_8, activation=a_tanh, name="Dense1",use_bias=True, bias_constraint=MinMaxNorm())(inputs)#tanh(S*x-t) 
x = tf.keras.activations.softmax(tf.matmul(x, tf.cast(K8, dtype=tf.float32), transpose_b=True))#softmax(K*tanh(S*x-t))
outputs = Dense(outputs_dim, activation="sigmoid",use_bias=False , name="Dense2")(x)#sigmod(softmax(K*tanh(S*x-t)))
model = Model(inputs = inputs, outputs = outputs)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
checkpoint = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='max')
model.fit(train_data, train_target, validation_split=0.33, batch_size=batch_size, epochs=10000, verbose=0, callbacks=[checkpoint])
end2 = time.time()
loss, acc = model.evaluate(test_data, test_target, batch_size=batch_size)
y_pre = model.predict(test_data)
#end2 = time.time()
print("time : ", end2-start2)
y_true = []
y_pred = []
for i in range(len(y_pre)):
    y_pred.append(np.argmax(y_pre[i]))
    y_true.append(np.argmax(test_target[i]))

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("accuracy_score :", accuracy_score(y_true, y_pred))
print("f1_score :", f1_score(y_true, y_pred))
print("precision_score :", precision_score(y_true, y_pred))
print("recall_score :", recall_score(y_true, y_pred))
#get the S and t : weight1 is S , bias1 is t , weight2 is v
weight1, bias1 = model.get_layer('Dense1').get_weights()
weight2 = model.get_layer('Dense2').get_weights()
#print(weight1, weight1.shape)
#print('\n')
#print(bias1, bias1.shape)
#print('\n')
#print(weight2[0], weight2[0].shape)

S = np.argmax(weight1, axis=0)
t1 = bias1
print(t1)
print(weight1)
node = []
"""
for i in range(len(S)):
    node.append(str(columns[IG[S[i]]])+ " < " + str(abs(t[i])))
dict = {node[0]:
        {node[1]:
         {node[3],node[4]},
         node[2]:
         {node[5],node[6]}
        }
       }
print(dict)
"""