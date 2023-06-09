# Sentiment Analysis with BERT

This is a Python project that uses the BERT model to perform sentiment analysis on text data. The project includes:

- Data cleaning and preprocessing using pandas and NLTK.
- Fine-tuning the pre-trained BERT model using TensorFlow and Hugging Face Transformers.
- Hyperparameter tuning using scikit-learn.
- Deploying the trained model to a web application using Flask.

# Installation

To run this project, you need to have Python 3.6 or later installed on your system, as well as the following packages:

- pandas
- nltk
- tensorflow
- transformers
- scikit-learn
- flask

You can install them using pip, for example:

`pip install pandas nltk tensorflow transformers scikit-learn flask`

# Usage
1. Clone or download this repository to your local machine.
2. Open a terminal and navigate to the project directory.
3. Run the following command to start the Flask application:

`python app.py`

4. Open a web browser and go to http://localhost:5000/predict. You should see a form where you can enter some text and get a sentiment prediction from the deployed BERT model.

# License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code as you see fit.
