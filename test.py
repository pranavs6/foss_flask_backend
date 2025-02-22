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
content_queue = []

@app.route('/check', methods=['POST'])
def check():
    data = request.json
    content_holder = data.get('text', '')
    mail_address_holder = data.get('mail_address', '')
    content_queue.append((mail_address_holder, content_holder))

    if len(content_queue) > 0:
        toxic = queue_handler()
    
    return jsonify({"toxic": toxic})

def classify(mail_address, content):
    inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    predictions = outputs.argmax(dim=-1).squeeze().tolist()
    result = {category: 'Toxic' if prediction > 0 else 'Not Toxic' for category, prediction in zip(categories, predictions)}
    
    toxic_flag = any(prediction == 'Toxic' for prediction in result.values())
    
    if toxic_flag:
        report(mail_address, result)
    else:
        store(mail_address, content)

    return toxic_flag

def report(mail_address, result):
    print("Offender found:", mail_address, "Categories:", result)

def store(mail_address, content):
    print("Safe person:", mail_address, "Content:", content)

def queue_handler():
    if content_queue:
        cur = content_queue.pop(0)
        return classify(cur[0], cur[1])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

