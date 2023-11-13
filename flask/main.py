import subprocess

#CONFIG
MODEL = "otisv1" #1 model as of now, keep it this


# List of required modules
required_modules = ["flask", "transformers", "torch", "numpy"]

# Check if each module is installed
for module in required_modules:
    try:
        __import__(module)
    except ImportError:
        print(f"{module} not found. Installing...")
        subprocess.check_call(["pip", "install", module])


#Check if we have OtisV1
import os
if os.path.exists(f"/{MODEL}"):
    os.mkdir(f"/{MODEL}")
    response = request.get("https://github.com/JewishLewish/Otis/raw/main/otisv1/model.safetensors")
    with open("/oti", 'wb') as file:
        file.write(response.content)

from flask import Flask, render_template, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

# Use the loaded model and tokenizer for inference
def get_prediction_with_loaded_model(text, loaded_model = AutoModelForSequenceClassification.from_pretrained('./otisv1/'), loaded_tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')):
    encoding = loaded_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    outputs = loaded_model(**encoding)

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(outputs.logits.squeeze().cpu()).detach().numpy()
    label = np.argmax(probs, axis=-1)

    if label == 1:
        return {
            'sentiment': 'Spam',
            'probability': probs[1]
        }
    else:
        return {
            'sentiment': 'Ham',
            'probability': probs[0]
        }


app = Flask(__name__)
app.secret_key = "put_something_here"

@app.route("/api", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        # Get parameter "Content" from the query string
        content_param = request.args.get("Content")
        return f"GET request - Content: {content_param}"

    elif request.method == "POST":
        # Get "Content" from JSON data in the request
        json_data = request.get_json()
        if json_data and "Content" in json_data:
            content_json = json_data["Content"]
            return f"POST request - Content: {content_json}"
        else:
            return "POST request - Content not found in JSON data"

if __name__ == "__main__":
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=8080)
    PORT = 8000
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)