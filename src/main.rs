#![feature(decl_macro)]

extern crate rocket;
//extern crate rocket_contrib;

use std::{collections::HashMap, process::{Stdio, Command}, io::{Write, Read}};
use rocket::{get, routes, Data};
use rocket_contrib::templates::Template;
use rusqlite::Connection;

struct DataSql {
    dbfile: String,
    connection: Connection,
}

impl DataSql {
    fn __init__(dbfile: &str) -> DataSql {
        let connection = Connection::open(dbfile).unwrap();
        let _ =connection.execute(
            
            r#"CREATE TABLE IF NOT EXISTS users ("email" TEXT, "name" TEXT, "business" TEXT, "id" INTEGER, "api_id" TEXT);"#,
            [],
        );

        DataSql { dbfile: dbfile.to_owned(), connection: connection }
    }
}

#[get("/")]
fn index() -> Template {
    let mut context = HashMap::new();
    context.insert("name", "Rocket!");
    return Template::render("index", &context);
}

use inline_python::python;

fn python() {
    let who = "world";
    let n = 5;
    python! {
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        import numpy as np

        # Use the loaded model and tokenizer for inference
        def get_prediction_with_loaded_model(text, loaded_model = AutoModelForSequenceClassification.from_pretrained("./model/"), loaded_tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")):
            encoding = loaded_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            outputs = loaded_model(**encoding)

            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(outputs.logits.squeeze().cpu()).detach().numpy()
            label = np.argmax(probs, axis=-1)

            if label == 1:
                return {
                    "sentiment": "Spam",
                    "probability": probs[1]
                }
            else:
                return {
                    "sentiment": "Ham",
                    "probability": probs[0]
                }


        def main(target):
            print(target)
            print(get_prediction_with_loaded_model(target))

        # Example usage:
        if __name__ == "__main__":
            import sys
            main("Hello world")
    }
}

fn main() {
   python();
}