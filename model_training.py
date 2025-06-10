# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense  # Dense = fully connected layer
from tensorflow.keras.models import Sequential  # Sequential = linear stack of layers
import pandas as pd  # For reading CSV data

# Function to define, compile, train, and save the neural network model
def neural_network(x, y):
    # Normalize the input features to improve training
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(x)  # Learn the mean and variance from the input data

    # Define the neural network architecture
    model = Sequential([
        normalizer,  # Add the normalization layer as the first input layer
        Dense(12, activation='relu', name='layer1'),   # First hidden layer with 12 units
        Dense(16, activation='relu', name='layer2'),   # Second hidden layer with 16 units
        Dense(8, activation='relu', name='layer3'),    # Third hidden layer with 8 units
        Dense(1, activation='sigmoid', name='layer4')  # Output layer (sigmoid for binary classification)
    ])

    # Compile the model with optimizer and loss function
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy()  # Since it's a binary classification
    )

    # Train the model on the dataset for 500 epochs
    model.fit(x, y, epochs=500)

    # Save the trained model to a file for future use
    model.save("mymodel.keras")

# Main function to load the data and call the training function
def main():
    # Load input features from CSV file
    input = pd.read_csv("training_input_dataset.csv")
    x = input.to_numpy()  # Convert to NumPy array

    # Load output labels from CSV file
    output = pd.read_csv("training_output_dataset.csv")
    y = output.to_numpy()  # Convert to NumPy array

    # Train the neural network model
    neural_network(x, y)

# Call the main function to start execution
main()
