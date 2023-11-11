#![feature(decl_macro)]

extern crate rocket;
//extern crate rocket_contrib;

use std::collections::HashMap;
use std::path::{PathBuf, Path};
use rocket::{get, routes};
use rocket_contrib::templates::Template;
use rusqlite::Connection;
use inline_python::{python, Context};

const DEBUG: bool = true;

/*
SQL CODE

Used to store user's data
_________________________
*/
#[warn(dead_code)]
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

/*
Website Code

Using Rust Rocket over Python Flask 
Python Flask has weird mem leaks -> https://stackoverflow.com/questions/49991234/flask-app-memory-leak-caused-by-each-api-call
______________________________________________________________________________________________________________________________
*/


use rocket::response::NamedFile;

#[get("/assets/<file..>")]
fn file(file: PathBuf) -> Option<NamedFile> {
    NamedFile::open(Path::new("assets/").join(file)).ok()
}


#[get("/")]
fn index() -> Template {
    let mut context = HashMap::new();
    context.insert("name", "Rocket!");
    return Template::render("index", &context);
}

#[get("/login")]
fn login() -> Template {
    let context: HashMap<String, String> = HashMap::new();
    return Template::render("login", &context);
}

#[get("/api")]
fn api() -> String {

    let c = python();
   
    c.run(python! {
        output = main("test")
    });
    return c.get::<String>("output");
}


/*
Python Inlining Code

Model I am using in Python
*/
fn python() -> Context {
    let c = Context::new();

    c.run (python! {
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
            return str(get_prediction_with_loaded_model(target))
    });

    return c;
}

fn main() {

   let data_sql = DataSql::__init__("users.db");
   rocket::ignite().attach(Template::fairing()).mount("/", routes![index, api, login, file]).launch();

   //rocket::ignite().attach(Template::fairing()).mount("/api", routes![api]).launch();
   //rocket::ignite().attach(Template::fairing()).mount("/login", routes![login]);
}