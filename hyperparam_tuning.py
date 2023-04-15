from sklearn.model_selection import GridSearchCV
from model import model, train_dataset

# Define hyperparameters to search
parameters = {
    "learning_rate": [1e-5, 2e-5, 3e-5],
    "batch_size": [16, 32],
    "epochs": [3, 4, 5]
}

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=3)
grid_search.fit(train_dataset)
