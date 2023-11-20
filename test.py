from transformers import pipeline


def analyze_output(input: str, PIPE = pipeline("text-classification", model="Titeiiko/OTIS-Official-Spam-Model")):
    x = PIPE(input)[0]
    if x["label"] == "LABEL_0":
        return {"type":"Not Spam", "probability":x["score"]}
    else:
        return {"type":"Spam", "probability":x["score"]}
    

if __name__ == "__main__":
    while(True):
        print(analyze_output(input(">>>")))