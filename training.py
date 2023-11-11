from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
from sklearn.model_selection import train_test_split
import torch
import numpy as np

def main():
    def process_data(row):
        #1 -> Spam
        text = row['sms']
        text = str(text)
        text = ' '.join(text.split())

        encodings = tokenizer(text, padding="max_length", truncation=True, max_length=128)

        label = 0
        if row["label"] == 1: #spam
            label = 1

        encodings['label'] = label
        encodings['text'] = text

        return encodings


    # Load the dataset
    #dataset = load_dataset('sms_spam')
    #columns_to_remove = ['message_id','text', 'label', 'date']

    #new_df = pd.read_csv("new_data.csv")

    # Convert to pandas DataFrame
    df = pd.read_csv("spam_dataset.csv")
    #df.drop(columns=columns_to_remove, inplace=True)


    # Save DataFrame to CSV
    #df.to_csv('spam_dataset.csv', index=False)


    print("Processing...")
    processed_data = []

    for i in range(len(df[:10000])):
        processed_data.append(process_data(df.iloc[i]))

    new_df = pd.DataFrame(processed_data)

    train_df, valid_df = train_test_split(
        new_df,
        test_size=0.2,
        random_state=2022
    )

    print("works")

    import pyarrow as pa
    from datasets import Dataset

    train_hg = Dataset(pa.Table.from_pandas(train_df))
    valid_hg = Dataset(pa.Table.from_pandas(valid_df))

    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        'google/bert_uncased_L-2_H-128_A-2',
        num_labels=2
    )

    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(output_dir="./result", evaluation_strategy="epoch",num_train_epochs=10)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hg,
        eval_dataset=valid_hg,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.evaluate()

    model.save_pretrained('./otisv1/')

    from transformers import AutoModelForSequenceClassification

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    new_model = AutoModelForSequenceClassification.from_pretrained('./otisv1/').to(device)

    from transformers import AutoTokenizer

    new_tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')

    def get_prediction(text):
        encoding = new_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

        outputs = new_model(**encoding)

        logits = outputs.logits
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sigmoid = torch.nn.Sigmoid()
        print(sigmoid)
        probs = sigmoid(logits.squeeze().cpu())
        probs = probs.detach().numpy()
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

    INPUT = """buy online and save viagra price for this high demand med best price for this high demand med best price for this high demand med buy nowbuy nowbuy price for this high demand med best price for this high demand med best price for this high demand med buy nowbuy nowbuy nowcialis soft price for this high demand med best price for this high demand med best price for this high demand med buy nowbuy nowbuy your penis width ( girth ) by 20 % gain up to 3 + full inches in length buy nowbuy now"""

    print(get_prediction(INPUT))

if __name__ == "__main__":
    main()