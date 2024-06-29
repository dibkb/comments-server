from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import uvicorn
from langdetect import detect, LangDetectException
from downloader import YoutubeCommentDownloader
import re
import asyncio

from transformers import pipeline

# Function to validate YouTube video IDs
def is_valid_youtube_id(video_id):
    regex = r'^[a-zA-Z0-9_-]{11}$'
    return bool(re.match(regex, video_id))


ml_models = {
    "sentiment_pipeline":None,
    "emotion_pipeline":None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    # Define paths for sentiment and emotion models
    model_path_sentiment = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    model_path_emotion = "SamLowe/roberta-base-go_emotions"

    ml_models["sentiment_pipeline"] = pipeline("sentiment-analysis", model=model_path_sentiment, tokenizer=model_path_sentiment, max_length=512, truncation=True)

    ml_models["emotion_pipeline"] = pipeline("text-classification", max_length=512, truncation=True, model=model_path_emotion, tokenizer=model_path_emotion)



    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)
# Define allowed origins for CORS
origins = [
    "http://localhost:3000",
    "https://commentsenseyt.vercel.app",
]

# Add CORS middleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize YouTube comment downloader
downloader = YoutubeCommentDownloader()

# Function to translate text to English
async def get_translated(text):
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
    # Make async request for translation
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, requests.get, url, params)
    translation = response.json()
    return ''.join([t['trans'] for t in translation['sentences']])

# Function to process a single comment
async def process_comment(comment):
    text = comment["text"]
    try:
        if detect(text) != 'en':
            comment['translated'] = await get_translated(text)
        else:
            comment['translated'] = False
    except LangDetectException:
        comment['translated'] = False
    
    # Perform sentiment and emotion analysis
    if comment['translated']:
        sentiment_task = asyncio.to_thread(ml_models['sentiment_pipeline'], comment['translated'])
        emotion_task = asyncio.to_thread(ml_models["emotion_pipeline"], comment['translated'])
    else:
        sentiment_task = asyncio.to_thread(ml_models["sentiment_pipeline"], text)
        emotion_task = asyncio.to_thread(ml_models["emotion_pipeline"], text)
    
    comment['sentiment'] = (await sentiment_task)[0]
    comment['emotion'] = (await emotion_task)[0]

    return comment

# Endpoint to fetch and process YouTube comments
@app.get("/video")
async def get_video(ytid: str, start: int, end: int, sort: int = Query(...)):
    # Validate YouTube video ID
    if not is_valid_youtube_id(ytid):
        raise HTTPException(status_code=404, detail="Invalid YouTube video ID")

    # Validate sort parameter
    if sort not in [0, 1]:
        raise HTTPException(status_code=404, detail="Invalid sort value. Expected either 0 or 1")
    # Initialize result list
    res = []
    # Fetch comments using downloader
    comments = downloader.get_comments(ytid, sort_by=sort)

    # Process comments concurrently
    tasks = []
    for i, comment in enumerate(comments):
        if start <= i < end:
            tasks.append(asyncio.create_task(process_comment(comment)))
        if i >= end:
            break

    res = await asyncio.gather(*tasks)
    return JSONResponse(res, media_type="application/json")

# Test endpoint to check if the server is running
@app.get("/test")
async def read_root():
    return {"message": "Hello World (powered by FastAPI) 🚀"}

# Main entry point to run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)