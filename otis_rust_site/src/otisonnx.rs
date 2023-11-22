use stable_inline_python::PyContext;

pub fn 
load(c: &PyContext) {

    let _x = c.run(r#"
from transformers import pipeline

def analyze_output(input: str, PIPE = pipeline("text-classification", model="Titeiiko/OTIS-Official-Spam-Model")):
    x = PIPE(input)[0]
    if x["label"] == "LABEL_0":
        return {"type":"Not Spam", "probability":x["score"]}
    else:
        return {"type":"Spam", "probability":x["score"]}
"#);
} 

pub fn 
run(c: &PyContext, input: &str) -> String{
    let _ = c.run(format!(r#"x = analyze_output("{}")"#,input).as_str());
    c.get::<String>("x").unwrap()
}