from transformers import pipeline
sentiment_pipeline = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
data = ["I think I like you but at the same time I don't. I don't know why."]
print(sentiment_pipeline(data))