import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score

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
  model.build(input_shape=(None, 12))
  model.fit(x , y , epochs = 500)
  return model

def main():
    df = pd.read_csv("sahi_dataset.csv")
    df = df.drop(columns=['patientid'])
    data = df.to_numpy()
    x = data[: , :-1]
    y = data[: , -1]
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    cv_scores = []
    for fold, (k_train_index, k_test_index) in enumerate(skf.split(x, y)):
        print(f"--- Running Fold {fold+1}/{N_SPLITS} ---")
        X_train, X_val = x[k_train_index], x[k_test_index]
        y_train, y_val = y[k_train_index], y[k_test_index]
        model = neural_network(X_train, y_train)
        model.save(f"model[{fold+1}].keras")
        predictions = model.predict(X_val)
        binary_predictions = (predictions > 0.5).astype(int)
        fold_accuracy = accuracy_score(y_val, binary_predictions)
        print(f"Fold {fold+1} Accuracy: {fold_accuracy}")
        cm = confusion_matrix(y_val, binary_predictions)
        print(f"Fold {fold+1} Confusion Matrix:\n{cm}")
        report = classification_report(y_val, binary_predictions, zero_division=0)
        print(f"Fold {fold+1} Classification Report:\n{report}")
        auc_score = roc_auc_score(y_val, predictions)
        print(f"Area Under the Curve (AUC): {auc_score:.4f}")
        cv_scores.append(fold_accuracy)
    avg_accuracy = np.mean(cv_scores)
    std_dev = np.std(cv_scores)
    print("\n==============================================")
    print(f"Average CV Accuracy ({N_SPLITS} folds): {avg_accuracy:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print("==============================================")
    print("\nTraining final production model on ALL data (1000 rows)...")
    final_model_production = neural_network(x, y) 
    final_model_production.save("mymodel.keras")

main()
