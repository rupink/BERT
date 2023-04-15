from data import clean_text
from model import model, train_dataset
from hyperparam_tuning import grid_search
from app import app

train_text = [...] # List of training texts
train_labels = [...] # List of training labels

train_text = [clean_text(text) for text in train_text]

best_params = grid_search.best_params_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']),loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=best_params['epochs'], batch_size=best_params['batch_size'])

if name == 'main':
  app.run(debug=True)
