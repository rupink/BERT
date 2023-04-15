from flask import Flask, request, jsonify
from transformers import BertTokenizer
from model import model

app = Flask(__name__)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    encoded_text = tokenizer.encode_plus(text, padding=True, truncation=True)
def make_prediction(input_text):
input_ids = tf.constant(encoded_text['input_ids'])
attention_mask = tf.constant(encoded_text['attention_mask'])
token_type_ids = tf.constant(encoded_text['token_type_ids'])
predictions = model.predict([input_ids, attention_mask, token_type_ids])
return predictions[0][0]

prediction = make_prediction(text)
if prediction >= 0.5:
    result = {'class': 'Positive', 'confidence': prediction}
else:
    result = {'class': 'Negative', 'confidence': 1 - prediction}
return jsonify(result)

if name == 'main':
  app.run(debug=True)
