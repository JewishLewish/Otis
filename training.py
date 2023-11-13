from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
import pyarrow as pa
from datasets import Dataset

def process_data(row, tokenizer):
    text = str(row['sms']).strip()
    encodings = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    label = 1 if row["label"] == 1 else 0
    encodings['label'] = label
    encodings['text'] = text
    return encodings

def load_and_process_data():
    df = pd.read_csv("spam_dataset.csv")
    processed_data = [process_data(df.iloc[i]) for i in range(min(len(df), 10000))]
    new_df = pd.DataFrame(processed_data)
    return new_df

def prepare_datasets(train_df, valid_size=0.2, random_state=2022):
    train_df, valid_df = train_test_split(train_df, test_size=valid_size, random_state=random_state)
    train_hg = Dataset(pa.Table.from_pandas(train_df))
    valid_hg = Dataset(pa.Table.from_pandas(valid_df))
    return train_hg, valid_hg

def train_and_evaluate(model, train_dataset, eval_dataset, tokenizer):
    training_args = TrainingArguments(output_dir="./result", evaluation_strategy="epoch", num_train_epochs=10)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.evaluate()

def save_model(model, path='./otisv1/'):
    model.save_pretrained(path)

def load_model_and_tokenizer(model_path='./otisv1/', tokenizer_name='google/bert_uncased_L-2_H-128_A-2'):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer

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

def main():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load and preprocess data
    new_df = load_and_process_data()

    # Prepare datasets
    train_hg, valid_hg = prepare_datasets(new_df)

    # Load or train the model
    model = AutoModelForSequenceClassification.from_pretrained('google/bert_uncased_L-2_H-128_A-2', num_labels=2)
    train_and_evaluate(model, train_hg, valid_hg, tokenizer)

    # Save the trained model
    save_model(model)

    # Load the model and tokenizer
    new_model, new_tokenizer = load_model_and_tokenizer()

    # Example prediction
    INPUT = """buy online and save... (your input text)"""
    print(get_prediction(new_model, new_tokenizer, INPUT)) 

if __name__ == "__main__":
    main()
