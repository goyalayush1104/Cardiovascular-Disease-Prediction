import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
import pandas as pd

def neural_network(x , y):
  normalizer = tf.keras.layers.Normalization(axis = -1)
  normalizer.adapt(x)
  model = Sequential([
      normalizer,
      Dense(12 , activation = 'relu' , name = 'layer1'),
      Dense(16 , activation = 'relu' , name = 'layer2'),
      Dense(8 , activation = 'relu' , name = 'layer3'),
      Dense(1 , activation = 'sigmoid' , name = 'layer4')
  ])
  model.compile(
      optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
      loss = tf.keras.losses.BinaryCrossentropy()
  )
  model.fit(x , y , epochs = 500)
  model.save("mymodel.keras")
  
def main():
    input = pd.read_csv("training_input_dataset.csv")
    x = input.to_numpy()
    output = pd.read_csv("training_output_dataset.csv")
    y = output.to_numpy()
    neural_network(x , y)

main()