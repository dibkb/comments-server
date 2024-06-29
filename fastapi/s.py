from itertools import islice
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import uvicorn
from langdetect import detect, LangDetectException
from downloader import YoutubeCommentDownloader
import re

from transformers import pipeline

model_path_sentiment = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model_path_emotion = "SamLowe/roberta-base-go_emotions"

sentiment_pipeline = pipeline("sentiment-analysis", model=model_path_sentiment, tokenizer=model_path_sentiment, max_length=512, truncation=True)

emotion_pipeline = pipeline("text-classification",max_length=512,truncation=True,model=model_path_emotion, tokenizer=model_path_emotion)
def is_valid_youtube_id(video_id):
    regex = r'^[a-zA-Z0-9_-]{11}$'
    return bool(re.match(regex, video_id))


app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://commentsenseyt.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
downloader = YoutubeCommentDownloader()


def get_translated(text):
    url = 'https://translate.googleapis.com/translate_a/single'
    params = {
        'client': 'gtx',
        'sl': 'auto',
        'tl': "en",
        'hl': "",
        'dt': ['t'],
        'dj': '1',
        'source': 'popup6',
        'q': text
    }
    translation= requests.get(url, params=params).json()
    return ''.join([t['trans'] for t in translation['sentences']])

def process_comment(comment):
    text = comment["text"]
    try:
        if detect(text) != 'en':
            comment['translated'] = get_translated(text)
        else:
            comment['translated'] = False
    except LangDetectException:
        comment['translated'] = False
    
    if comment['translated']:
        comment['sentiment'] = sentiment_pipeline(comment['translated'])[0]
        comment['emotion'] = emotion_pipeline(comment['translated'])[0]
    else:
        # Run sentiment and emotion tasks concurrently
        comment['sentiment'] = sentiment_pipeline(text)[0]
        comment['emotion'] = emotion_pipeline(text)[0]

    return comment

@app.get("/sync-video")
async def get_video(ytid: str, start: int, end: int, sort: int = Query(...)):
        # Handle Youtubeid
    if(is_valid_youtube_id(ytid) == False):
        raise HTTPException(status_code=404, detail="Invalid YouTube videoid")

    # Handle sort parameters
    if sort != 0 and sort != 1:
        raise HTTPException(status_code=404, detail="Invalid sort value. Expected either 0 or 1")
    res = []
    comments = downloader.get_comments(ytid, sort_by=sort)

    for c in islice(comments,start,end):
        res.append(process_comment(c))
    return JSONResponse(res, media_type="application/json")

@app.get("/test")
async def read_root():
    return {"message": "Hello World (powered by FastAPI) 🌍"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=9000, reload=True)
