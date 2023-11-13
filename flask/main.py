import subprocess

import requests

#CONFIG
MODEL = "otisv1" #1 model as of now, keep it this

#Colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def install_needed_modules():
    print(f"[{bcolors.WARNING}Module Not Founded{bcolors.ENDC}] -> Didn't find certain module needed")
    
    required_modules = ["flask", "transformers", "torch", "numpy"]

    # Check if each module is installed
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            print(f"[{bcolors.WARNING}Module Not Founded{bcolors.ENDC}] -> {module} not found. Installing...")
            subprocess.check_call(["pip", "install", module])
            print(f"[{bcolors.OKGREEN}{module} Successfully Downloaded{bcolors.ENDC}] -> Successfully downloaded latest version for module: {module}")

try:
    from flask import Flask, render_template, request
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    import numpy as np
except:
    install_needed_modules()


def download_model():
    import os

    print(f"[{bcolors.WARNING}Model Not Found{bcolors.ENDC}] -> Otis Anti-Spam AI Model is not found... Installing...")
    os.makedirs(MODEL, exist_ok=True)

    # Download model.safetensors
    response = requests.get(f"https://github.com/JewishLewish/Otis/raw/main/{MODEL}/model.safetensors")
    with open(f"{MODEL}/model.safetensors", 'wb') as file:
        file.write(response.content)
        print(f"[{bcolors.OKGREEN}{MODEL} Successfully Downloaded{bcolors.ENDC}] -> Found the model you are looking for!")

    # Download config.json
    response = requests.get(f"https://github.com/JewishLewish/Otis/raw/main/{MODEL}/config.json")
    with open(f"{MODEL}/config.json", 'wb') as file:
        file.write(response.content)
        print(f"[{bcolors.OKGREEN}{MODEL} config.json FOUND{bcolors.ENDC}] -> Found the model config.json you are looking for!")


try:
    _ = AutoModelForSequenceClassification.from_pretrained('./otisv1/')
    del _
except:
    download_model()


exit()



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