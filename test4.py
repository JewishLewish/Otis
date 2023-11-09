from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

# Load the saved model
loaded_model = AutoModelForSequenceClassification.from_pretrained('./model/')

# Load the tokenizer associated with the pre-trained model
loaded_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Use the loaded model and tokenizer for inference
def get_prediction_with_loaded_model(text):
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
while True:
    print("User input:")
    x = input()
    result = get_prediction_with_loaded_model(x)
    print(result)
