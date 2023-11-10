#![feature(decl_macro)]

extern crate rocket;
//extern crate rocket_contrib;

use std::collections::HashMap;
use rocket::{get, routes};
use rocket_contrib::templates::Template;
use rusqlite::Connection;

struct DataSql {
    dbfile: String,
    connection: Connection,
}

impl DataSql {
    fn new(dbfile: &str) -> &str {
        let connection = Connection::open(dbfile).unwrap();
        let _ =connection.execute(
            
            r#"CREATE TABLE IF NOT EXISTS users ("email" TEXT, "name" TEXT, "business" TEXT, "id" INTEGER);"#,
            [],
        );
        return "Works";
    }
}

#[get("/")]
fn index() -> Template {
    let mut context = HashMap::new();
    context.insert("name", "Rocket!");
    return Template::render("index", &context);
}

fn main() {
    let mut data_sql = DataSql::new("users.db");
    //rocket::ignite().attach(Template::fairing()).mount("/", routes![index]).launch();
}