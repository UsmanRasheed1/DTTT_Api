from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
import os

# Load smaller model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Label mapping for SST-2 (2 labels)
label_map = {
    0: "Negative",
    1: "Positive"
}

def classify_distilbert_sentiment(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        logits = model(**encoded_input).logits

    probs = F.softmax(logits, dim=1)[0]

    top_label = torch.argmax(probs).item()
    sentiment = label_map[top_label]

    return sentiment

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Hello, world!"

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No 'message' provided."}), 400

    sentiment = classify_distilbert_sentiment(message)
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port)
