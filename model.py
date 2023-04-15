import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification

# Load pre-trained BERT model
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4", trainable=True)

# Define BERT model architecture
max_length = 128
input_word_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_mask")
input_type_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_type_ids")
bert_inputs = {"input_word_ids": input_word_ids, "input_mask": input_mask, "input_type_ids": input_type_ids}
pooled_output, _ = bert_layer(bert_inputs)
dense_layer = tf.keras.layers.Dense(1, activation="sigmoid")(pooled_output)
model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=dense_layer)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Load the data
train_text = [    "I loved the movie, it was amazing!",    "The acting was terrible, I couldn't even finish the movie",    "The plot was too predictable, I expected more",    "The special effects were mind-blowing, I was on the edge of my seat",    "The characters were flat and uninteresting, I didn't care what happened to them",    "The dialogue was witty and engaging, I laughed out loud several times",    "The pacing was slow and boring, I kept checking my phone",    "The cinematography was stunning, every shot was a work of art",    "The soundtrack was distracting and didn't fit the mood of the movie",    "The ending was satisfying and tied up all the loose ends",    "The movie was a complete waste of time, I regret watching it",    "The movie was an instant classic, I will definitely watch it again"] # List of training texts
train_labels = [
    1,  # Positive
    0,  # Negative
    0,  # Negative
    1,  # Positive
    0,  # Negative
    1,  # Positive
    0,  # Negative
    1,  # Positive
    0,  # Negative
    1,  # Positive
    0,  # Negative
    1   # Positive
] # List of training labels
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
train_encoded = tokenizer(train_text, padding=True, truncation=True, max_length=max_length, return_tensors='tf')
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encoded), train_labels)).shuffle(len(train_labels)).batch(32)

# Train the model
model.fit(train_dataset, epochs=5)
