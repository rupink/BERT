import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import layers

# Load the pre-trained BERT model from TensorFlow Hub
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3", trainable=True)

# Define the model architecture
input_text = layers.Input(shape=(), dtype=tf.string)
bert_output = bert_layer(input_text)["pooled_output"]
dense = layers.Dense(1, activation="sigmoid")(bert_output)
model = tf.keras.models.Model(inputs=input_text, outputs=dense)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Load the movie review dataset
df = pd.read_csv("movie_reviews.csv")

# Clean the text data
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["text"] = df["text"].apply(clean_text)

# Split the dataset into training and validation sets
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

# Tokenize the text data
tokenizer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
train_text = train_df["text"].tolist()
train_text = [str(text) for text in train_text]
train_labels = train_df["label"].tolist()
train_labels = [int(label) for label in train_labels]
train_encoded = tokenizer(train_text)
train_dataset = tf.data.Dataset.from_tensor_slices((train_encoded['input_word_ids'], train_encoded['input_mask'], train_encoded['input_type_ids'], train_labels)).shuffle(10000).batch(32)

val_text = val_df["text"].tolist()
val_text = [str(text) for text in val_text]
val_labels = val_df["label"].tolist()
val_labels = [int(label) for label in val_labels]
val_encoded = tokenizer(val_text)
val_dataset = tf.data.Dataset.from_tensor_slices((val_encoded['input_word_ids'], val_encoded['input_mask'], val_encoded['input_type_ids'], val_labels)).batch(32)

#Fine-tune the BERT model on our dataset
history = model.fit(train_dataset, epochs=5, validation_data=val_dataset)

#Perform hyperparameter tuning to find the optimal hyperparameters
parameters = {
"learning_rate": [1e-5, 2e-5, 3e-5],
"batch_size": [16, 32],
"epochs": [3, 4, 5]
}
grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=3)
grid_search.fit(train_dataset)

#Deploy the model to a web application using Flask
from flask import Flask, request, jsonify

#app = Flask(name)

@app.route('/predict', methods=['POST'])
def predict():
  text = request.json['text']
  encoded_text = tokenizer([text])
  prediction = model.predict(encoded_text)[0][0]
  if prediction >= 0.5:
    sentiment = "Positive"
  else:
    sentiment = "Negative"
  response = {
  'sentiment': sentiment,
  'prediction': float(prediction)
  }
  return jsonify(response)

if name == 'main':
  app.run(debug=True)
