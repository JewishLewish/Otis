#![feature(decl_macro)]

extern crate rocket;
//extern crate rocket_contrib;

use std::collections::HashMap;
use std::path::{PathBuf, Path};
use rocket::request::Form;
use rocket::{get, routes, FromForm, post};
use rocket_contrib::templates::Template;
use rusqlite::Connection;
use inline_python::{python, Context};


/* 
CONSTANT VARIABLES
Doesn't change; Consistent Throughout Code
*/
const DB_FILE: &str = "user.db";

/*
SQL CODE

Used to store user's data
_________________________
*/
#[warn(dead_code)]
struct DataSql {
    dbfile: String
}

struct UserSql {
    email: String,
    name: String,
    business: String,
    id: i32,
    api_id: i64
}

impl Default for DataSql {
    fn default() -> Self {
        DataSql { dbfile: DB_FILE.to_string()}
    }
}

impl DataSql {
    
    fn __init__() {
        let connection = Connection::open(DB_FILE).unwrap();
        let _ =connection.execute(
            
            r#"CREATE TABLE IF NOT EXISTS users ("email" TEXT, "name" TEXT, "business" TEXT, "id" INTEGER, "api_id" TEXT);"#,
            [],
        );
    }

    fn add_user(self, input: &Login) -> bool {
        let connection = Connection::open(self.dbfile.to_owned()).unwrap();
        if !(self.user_exist(input.email.to_owned())) {
            return false;
        } else {
            
            let u_id = &self.unique_id();
            let _ =connection.execute(
            
            format!(r#"INSERT INTO users (email, name, business, id, api_id) VALUES ("{}", "{}", "{}", {}, "{}");"#, input.email, input.password, input.business, u_id, &self.api_unique_id(input, &u_id)).as_str(),
            [],
            );

            return true;
        }
    }

    fn user_exist(&self, email: String) -> bool {
        let connection = Connection::open(&self.dbfile).unwrap();

        let count: i8 = connection.query_row(
            format!("SELECT COUNT(*) FROM users WHERE email = '{}'",email).as_str(),
            [],
            |row| row.get(0),
        ).expect("Failed to get row count");

        if count == 0 { true } else { false }
    }

    fn unique_id(&self) -> u32 {
        let connection = Connection::open(&self.dbfile).unwrap();
        
        let count: u32 = connection.query_row(
            format!("SELECT COUNT(*) FROM users").as_str(),
            [],
            |row| row.get(0),
        ).expect("Failed to get row count");

        return count;
    }

    fn api_unique_id(&self, input: &Login, u_id: &u32) -> String {

        format!("{}+{}+{}", input.business, input.password, u_id )
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

#[derive(FromForm, Debug)]
struct Login {
    email: String,
    password: String,
    business: String
}

#[post("/login", format = "application/x-www-form-urlencoded", data = "<user_input>")]
fn loginpost(user_input: Form<Login>) -> String {

    let copy = user_input.0;
    let success = DataSql::add_user(DataSql {..Default::default()} , &copy);

    if success {
        format!("Successfully Inserted {:?}", copy)
    } else {
        format!("FAILED!")
    }
    

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

   DataSql::__init__();
   
   //let x = DataSql::add_user(login { email: "test@gmail.com".to_string(), content: "Password".to_string(), business: "Mewgem".to_string() });
   //print!("{}",x);

   rocket::ignite().attach(Template::fairing()).mount("/", routes![index, api, login, loginpost, file]).launch();

   //rocket::ignite().attach(Template::fairing()).mount("/api", routes![api]).launch();
   //rocket::ignite().attach(Template::fairing()).mount("/login", routes![login]);
}