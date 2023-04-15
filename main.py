from data import clean_text
from model import model, train_dataset
from hyperparam_tuning import grid_search
from app import app

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

train_text = [clean_text(text) for text in train_text]

best_params = grid_search.best_params_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']),loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=best_params['epochs'], batch_size=best_params['batch_size'])

if name == 'main':
  app.run(debug=True)
