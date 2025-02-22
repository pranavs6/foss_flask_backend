from flask import Flask, request, jsonify
from transformers import DebertaV2Tokenizer
from celadon.model import MultiHeadDebertaForSequenceClassification
import torch
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
app = Flask(__name__)

# Load tokenizer and model
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

    # Handle the queue and get toxicity result
    if len(content_queue) > 0:
        toxic = queue_handler()
    
    return jsonify({"toxic": toxic})

def classify(mail_address, content):
    inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    predictions = outputs.argmax(dim=-1).squeeze().tolist()
    result = {category: 'Toxic' if prediction > 0 else 'Not Toxic' for category, prediction in zip(categories, predictions)}
    
    # Determine if any category is toxic
    toxic_flag = any(prediction == 'Toxic' for prediction in result.values())
    
    # Send email if toxic, else store content
    if toxic_flag:
        report(mail_address, content, result)
    else:
        store(mail_address, content)

    return toxic_flag

def store(mail_address, content):
    print("Safe person:", mail_address, "Content:", content)

def report(mail_address, content, result):
    # Prepare email content
    toxic_categories = [category for category, status in result.items() if status == 'Toxic']
    email_body = f"""Dear user,

Your comment was flagged for containing toxic content in the following category/categories: {', '.join(toxic_categories)}.

Comment: "{content}"

Please review your comment.

Best regards,
OSHUB Team"""

    # Gmail SMTP credentials
    gmail_user = "mailer.oshub@gmail.com"  # Replace with your Gmail email
    gmail_password = "vfry lzvq okzb ainc"    # Replace with the App Password from Google

    # Create the email message
    message = MIMEMultipart()
    message["From"] = gmail_user
    message["To"] = mail_address
    message["Subject"] = "Toxic Content Warning"
    message.attach(MIMEText(email_body, "plain"))

    try:
        # Connect to Gmail SMTP server
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()  # Secure the connection
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, mail_address, message.as_string())
        server.quit()
        print(f"Email sent to: {mail_address}")
    except Exception as e:
        print(f"Error sending email: {e}")

def queue_handler():
    # Process the first item in the queue
    if content_queue:
        cur = content_queue.pop(0)
        return classify(cur[0], cur[1])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

