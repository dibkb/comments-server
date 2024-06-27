from transformers import pipeline

model_path_sentiment = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model_path_emotion = "SamLowe/roberta-base-go_emotions"

sentiment_pipeline = pipeline("sentiment-analysis", model=model_path_sentiment, tokenizer=model_path_sentiment, max_length=512, truncation=True)

emotion_pipeline = pipeline("text-classification",max_length=512,truncation=True,model=model_path_emotion, tokenizer=model_path_emotion)
