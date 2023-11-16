try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    import numpy as np
except:
    print("FUCK")

def get_prediction(model, tokenizer, text):
    encoding = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    encoding = {k: v.to(model.device) for k, v in encoding.items()}
    outputs = model(**encoding)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(outputs.logits.squeeze().cpu()).detach().numpy()
    label = np.argmax(probs, axis=-1)
    
    return {
        'sentiment': 'Spam' if label == 1 else 'Ham',
        'probability': probs[1] if label == 1 else probs[0]
    }

def load_model_and_tokenizer(model_path=f'./otisv1/', tokenizer_name='google/bert_uncased_L-2_H-128_A-2'):
    # Load the model's state_dict using torch.load
    model_state_dict = torch.load(f"{model_path}/model.pth")
    model = AutoModelForSequenceClassification.from_pretrained(tokenizer_name, state_dict=model_state_dict)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

MODEL, TOKENIZER = load_model_and_tokenizer()

# Example usage:
if __name__ == "__main__":
    x = get_prediction(MODEL, TOKENIZER, "test")
    print(x)