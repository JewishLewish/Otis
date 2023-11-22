mod otisonnx;

#[macro_use] extern crate rocket;
//extern crate rocket_contrib;

use std::collections::HashMap;
use std::path::{PathBuf, Path};
use otisonnx::{load, run};
use rocket::form::Form;
use rocket::{get, routes, FromForm, post, uri};
use rocket_dyn_templates::Template;
use rusqlite::Connection;
use rand::Rng;


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

#[derive(Debug)]
struct UserSql {
    email: String,
    password: String,
    business: String,
    id: i32,
    api_id: String
}

impl Default for DataSql {
    fn default() -> Self {
        DataSql { dbfile: DB_FILE.to_string()}
    }
}

impl DataSql {
    /// Initializes the database if it doesn't exist.
    fn __init__() {
        let connection = Connection::open(DB_FILE).unwrap();
        let _ =connection.execute(
            
            r#"CREATE TABLE IF NOT EXISTS users ("email" TEXT, "password" TEXT, "business" TEXT, "id" INTEGER, "api_id" TEXT);"#,
            [],
        );
    }

    /// Adds a new user to the database.
    ///
    /// # Arguments
    ///
    /// * `input` - A reference to a `Login` struct containing user account details.
    ///
    /// # Returns
    ///
    /// Returns `true` if the user was successfully added, `false` if the user already exists.
    fn add_user(self, input: &Register) -> bool {
        
        let connection = Connection::open(self.dbfile.to_owned()).unwrap();
        if self.user_exist(input.email.to_owned()) {
            return false;
        } else {
            
            let u_id = &self.generate_unique_id();
            let _ =connection.execute(
            
            format!(r#"INSERT INTO users (email, password, business, id, api_id) VALUES ("{}", "{}", "{}", {}, "{}");"#, input.email, input.password, input.business, u_id, &self.generate_api_unique_id(input, &u_id)).as_str(),
            [],
            );

            return true;
        }
    }

    /// Checks if a user with the given email already exists in the database.
    ///
    /// # Arguments
    ///
    /// * `email` - A String representing the email address of the user.
    ///
    /// # Returns
    ///
    /// Returns `true` if the user exists, `false` otherwise.
    fn user_exist(&self, email: String) -> bool {
        let connection = Connection::open(&self.dbfile).unwrap();

        let count: u8 = connection.query_row(
            format!("SELECT COUNT(*) FROM users WHERE email = '{}'",email).as_str(),
            [],
            |row| row.get(0),
        ).expect("Failed to get row count");

        if count == 0 { false } else { true }
        // false -> no acct associated / does not exists with email
        // true -> acct associated / exists with email
    }

    fn token_exist(&self, api_token: String) -> bool {
        let connection = Connection::open(&self.dbfile).unwrap();

        let count: u8 = connection.query_row(
            format!("SELECT COUNT(*) FROM users WHERE api_id = '{}'",api_token).as_str(),
            [],
            |row| row.get(0),
        ).expect("Failed to get row count");

        if count == 0 { false } else { true }
        // false -> no acct associated / THERE IS NO TOKEN WITH SAME ID
        // true -> acct associated / THIS IS TOKEN WITH SAME ID
    }

    /// Generates a unique ID based on the current number of users in the database.
    ///
    /// # Returns
    ///
    /// Returns a unique ID as a `u32`.
    fn generate_unique_id(&self) -> u32 {
        let connection = Connection::open(&self.dbfile).unwrap();
        
        let count: u32 = connection.query_row(
            format!("SELECT COUNT(*) FROM users").as_str(),
            [],
            |row| row.get(0),
        ).expect("Failed to get row count");

        return count;
    }

    /// Generates a unique API ID based on the user's business, password, and unique ID.
    ///
    /// # Arguments
    ///
    /// * `input` - A reference to a `Login` struct containing user account details.
    /// * `u_id` - A reference to the user's unique ID.
    ///
    /// # Returns
    ///
    /// Returns a unique API ID as a `String`.
    fn generate_api_unique_id(&self, input: &Register, u_id: &u32) -> String {

        let mut rng = rand::thread_rng();

        let random_number: u32 = rng.gen();

        format!("{}{}_{}{}", input.email, input.password, u_id, random_number)
        //to ensure encryption it uses:
        //  email -> since one email can usually be assigned to a person
        //  password -> user picks their own password
        //  u_id -> each user have their own unique id (similiar to how each user have their own email)
        //  random_number -> each user has a generated random number (u32) to ensure a level of security
    }

    fn find_data_with_email(&self, email: String) -> Option<UserSql> {
        let conn = Connection::open(&self.dbfile).unwrap();

        let mut stmt = conn
            .prepare(&format!(r#"SELECT email, password, business, id, api_id FROM users WHERE email = '{}'"#, email).to_string())
            .unwrap();

        let mut rows = stmt.query([]).unwrap();

        if let Some(row) = rows.next().unwrap() {
            Some(UserSql {
                email: row.get(0).unwrap(),
                password: row.get(1).unwrap(),
                business: row.get(2).unwrap(),
                id: row.get(3).unwrap(),
                api_id: row.get(4).unwrap(),
            })
        } else {
            None
        }
    }
}

/*
Website Code

Using Rust Rocket over Python Flask 
Python Flask has weird mem leaks -> https://stackoverflow.com/questions/49991234/flask-app-memory-leak-caused-by-each-api-call
______________________________________________________________________________________________________________________________
*/
use rocket::response::Redirect;
use rocket::fs::NamedFile;

/// every item in the /assets/ folder can be used
/// This can include .css files, .img files, etc.
#[get("/assets/<file..>")]
async fn file(file: PathBuf) -> Option<NamedFile> {
    match NamedFile::open(Path::new("assets/").join(file)).await {
        Ok(named_file) => Some(named_file),
        Err(_) => None,
    }
}

use rocket::http::{Cookie, CookieJar};

/// home page
#[get("/")]
fn index(cookies: &CookieJar<'_>) -> Result<Template, Redirect> {
    if let Some(_) = cookies.get("token") {
        // Token is present, redirect to apitoken route
        return Err(Redirect::to(uri!(apitoken)));
    }

    print!("NO COOKIES FOUND!");
    let context: HashMap<String, String> = HashMap::new();
    Ok(Template::render("index", &context))
}


/// This is for loginpost(); Registering an Account
/// Contains: email, password and business associated
#[derive(FromForm, Debug)]
struct Register {
    email: String,
    password: String,
    business: String
}

/// Gets Register page 
#[get("/register")]
fn register() -> Template {
    let context: HashMap<String, String> = HashMap::new();
    return Template::render("register", &context);
}

/// Processes User Login Information
/// Checks if User account already exists or not
/// If exists -> Do not Insert Data to SQL
/// Otherwise -> Insert data to SQL; Generate Unique_ID , API_Unique_ID
#[post("/register", format = "application/x-www-form-urlencoded", data = "<user_input>")]
fn registerpost(user_input: Form<Register>, cookies: &CookieJar<'_>) -> Redirect {

    print!("{:?}",user_input);
    let copy = user_input.into_inner();

    //let copy = user_input.0;
    let success = DataSql::add_user(DataSql {..Default::default()} , &copy);


    if success {
        let x = DataSql::find_data_with_email(&DataSql {..Default::default()}, copy.email).unwrap();
        cookies.add(Cookie::new("token", x.api_id));
        cookies.add(Cookie::new("email", x.email));

        Redirect::to(uri!(apitoken))
    } else {
        Redirect::to(uri!(register))
    }
}

#[derive(Serialize, FromForm, Debug)]
struct Login {
    email: String,
    password: String,
}

#[get("/login")]
fn login() -> Template {
    let context: HashMap<String, String> = HashMap::new();
    return Template::render("login", &context);
}


#[post("/login", format = "application/x-www-form-urlencoded", data = "<user_input>")]
fn loginpost(user_input: Form<Login>, cookies: &CookieJar<'_>) -> Redirect {

    print!("{:?}",user_input);

    let copy = user_input.into_inner();

    let success = DataSql::user_exist(&DataSql {..Default::default()} , copy.email.to_owned());

    if success {
        let x = DataSql::find_data_with_email(&DataSql {..Default::default()}, copy.email).unwrap();
        cookies.add(Cookie::new("token", x.api_id));
        cookies.add(Cookie::new("email", x.email));

        Redirect::to(uri!(apitoken))
    } else {
        Redirect::to(uri!(login))
    }
}

/// Api Page
#[get("/api?<content>")]
fn api(mut content: String, cookies: &CookieJar<'_>) -> String {

    if cookies.get("token").map(|crumb| format!("{}", crumb.value())).is_none() { //variable exists
        return r#"{"Status": "No Token Associated"}"#.to_string();
    }

    let email = cookies.get("email").map(|crumb| format!("{}", crumb.value()));

    if !(DataSql::user_exist(&DataSql { ..Default::default() }, email.unwrap())) {
        return r#"{"Status": "No Email Associated"}"#.to_string();
    }

    let api_token = cookies.get("token").unwrap().value();

    if !(DataSql::token_exist(&DataSql { ..Default::default() }, api_token.to_string())) {
        return r#"{"Status": "No Token Associated"}"#.to_string();
    }

    //all checks are passed!

    content = content.replace("%20", " ");

    let c = PyContext::new();
    load(&c);
    let x  = run(&c, &content);

    format!("{}",x)
}


/// API token for user
#[get("/apitoken")]
fn apitoken(cookies: &CookieJar<'_>) -> Result<Redirect, Template> {
    if let Some(_) = cookies.get("token") {
        let email = cookies.get("email").unwrap();

        let x = DataSql::find_data_with_email(&DataSql {..Default::default()}, email.value().to_string()).unwrap();


        let mut context = HashMap::new();
        context.insert("user", x.email);
        context.insert("api_token", x.api_id);
        return Err(Template::render("api_token", &context));
    }

    Ok(Redirect::to(uri!(index)))
}
use serde::Deserialize;
#[derive(Serialize, Debug, PartialEq, Eq, Deserialize)]
#[serde(crate = "rocket::serde")]
struct ApiInput {
    content: String,
    api_token: String
}

use rocket::serde::{Serialize, json::Json};
use stable_inline_python::PyContext;


#[post("/api", format = "application/json", data = "<json_data>")]
fn api_post(json_data: Json<ApiInput>) -> String {

    let json_data =  json_data.into_inner();

    let api_token = json_data.api_token;

    if !(DataSql::token_exist(&DataSql { ..Default::default() }, api_token.to_string())) {
        return r#"{"Status": "No Api Token Associated"}"#.to_string();
    }

    let c = PyContext::new();
    load(&c);
    let x  = run(&c, &json_data.content);

    format!("{}",x)
}



/// Python Inlining Code
/// Allows for Python Bindings in Rust
/* fn python() -> Context {
    let c = Context::new();

    c.run (python! {
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        import numpy as np

        # Use the loaded model and tokenizer for inference
        def get_prediction_with_loaded_model(text, loaded_model = AutoModelForSequenceClassification.from_pretrained("./otisv1/"), loaded_tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")):
            encoding = loaded_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            outputs = loaded_model(**encoding)

            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(outputs.logits.squeeze().cpu()).detach().numpy()
            label = np.argmax(probs, axis=-1)

            if label == 1:
                return {
                    "type": "Spam",
                    "probability": probs[1]
                }
            else:
                return {
                    "type": "Ham",
                    "probability": probs[0]
                }


        def main(target):
            return str(get_prediction_with_loaded_model(target))
    });

    return c;
} */



#[launch]
fn rocket() -> _ {

    //let c = PyContext::new();

    //load(&c);
    //let x  = run(&c, "test");
    //print!("{}",x);

   DataSql::__init__();
   //let x = DataSql::find_data_with_email(&DataSql {..Default::default()}, "test".to_string());
   //print!("{}",x.unwrap().email);
   
   //let x = DataSql::add_user(login { email: "test@gmail.com".to_string(), content: "Password".to_string(), business: "Mewgem".to_string() });
   //print!("{}",x);

   rocket::build().mount("/", routes![index, api, register, registerpost, file, login, loginpost, apitoken, api_post]).attach(Template::fairing())
}