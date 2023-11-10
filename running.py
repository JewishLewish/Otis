from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

# Use the loaded model and tokenizer for inference
def get_prediction_with_loaded_model(text, loaded_model = AutoModelForSequenceClassification.from_pretrained('./model/'), loaded_tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')):
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


def main(*argv):
    combined_args = ' '.join(argv)
    print(get_prediction_with_loaded_model(combined_args))

# Example usage:
if __name__ == "__main__":
    import sys
    arguments = sys.argv[1:]
    main(*arguments)