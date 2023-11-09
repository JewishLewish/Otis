from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

def loadmodel():
    return AutoModelForSequenceClassification.from_pretrained('./model/')

def load_tokenizer():
    return AutoTokenizer.from_pretrained('bert-base-uncased')

# Use the loaded model and tokenizer for inference
def get_prediction_with_loaded_model(text, loaded_model = loadmodel(), loaded_tokenizer = load_tokenizer()):
    encoding = loaded_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    outputs = loaded_model(**encoding)

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(outputs.logits.squeeze().cpu()).detach().numpy()
    label = np.argmax(probs, axis=-1)
    print(label)

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

# Example usage:
if __name__ == "__main__":
    while True:
        print("User input:")
        x = input()
        result = get_prediction_with_loaded_model(x)
        print(result)
