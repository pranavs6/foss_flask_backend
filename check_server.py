from flask import Flask, request, jsonify
from transformers import DebertaV2Tokenizer
from celadon.model import MultiHeadDebertaForSequenceClassification
import torch

app = Flask(__name__)

tokenizer = DebertaV2Tokenizer.from_pretrained("/home/pranavsathyaar/main/works/oshub_flask/celadon")
print("Tokenizer loaded")
model = MultiHeadDebertaForSequenceClassification.from_pretrained("/home/pranavsathyaar/main/works/oshub_flask/celadon")
model.eval()

categories = ['Race/Origin', 'Gender/Sex', 'Religion', 'Ability', 'Violence']

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    sample_text = data.get('text', '')

    inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    predictions = outputs.argmax(dim=-1).squeeze().tolist()
    result = {category: 'Toxic' if prediction > 0 else 'Not Toxic' for category, prediction in zip(categories, predictions)}

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 


