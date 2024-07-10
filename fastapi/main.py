from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import uvicorn
from langdetect import detect, LangDetectException
from downloader import YoutubeCommentDownloader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from make_db import make_vector_database, get_description
import re
import asyncio
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from transformers import pipeline
from prompt import PROMPT_TEMPLATE
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
class ChatRequest(BaseModel):
    video_id: str
    query: str
class PrepareRequest(BaseModel):
    video_id: str

engine = create_engine("sqlite:///video_info.db")
Session = sessionmaker(bind=engine)
Base = declarative_base()

class VideoInfo(Base):
    __tablename__ = "video_info"

    id = Column(Integer, primary_key=True)
    video_id = Column(String, unique=True)
    title = Column(String)
    description = Column(String)
    channel_name = Column(String)
    channel_url = Column(String)
    channel_subscribers = Column(String)
    duration = Column(Integer)
    views = Column(Integer)

Base.metadata.create_all(engine)

@app.post("/prepare")
async def prepare(request: PrepareRequest):
    session = Session()
    db_video_info = session.query(VideoInfo).filter_by(video_id=request.video_id).first()
    make_vector_database(request.video_id)
    if db_video_info is not None:
        return {"status": "success"}

    video_info = get_description(request.video_id)
    db_video_info = VideoInfo(
        video_id=request.video_id,
        title=video_info['title'],
        description=video_info['shortDescription'],
        channel_name=video_info['channel']['name'],
        channel_url=video_info['channel']['url'],
        channel_subscribers=video_info['channel']['subscribers']['pretty'],
        duration=video_info['duration']['lengthSec'],
        views=video_info['views']['pretty']
    )
    session.add(db_video_info)
    session.commit()
    session.close()

    return {"status": "success"}


@app.post("/chat")
async def chat(request: ChatRequest):
    session = Session()
    db_video_info = session.query(VideoInfo).filter_by(video_id=request.video_id).first()
    session.close()

    if db_video_info is None:
        return JSONResponse({"content":"Video information not found."}, media_type="application/json")

    video_info = {
        'title': db_video_info.title,
        'shortDescription': db_video_info.description,
        'channel': {
            'name': db_video_info.channel_name,
            'url': db_video_info.channel_url,
            'subscribers': {
                'pretty': db_video_info.channel_subscribers
            }
        },
        'duration': {
            'lengthSec': db_video_info.duration
        },
        'views': {
            'pretty': db_video_info.views
        }
    }

    CHROMA_PATH = f'data/chroma_db_{request.video_id}'
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.3)

    results = db.similarity_search_with_relevance_scores(request.query, k=3)

    context_text = "\n\n".join([doc.page_content for doc, _score in results])

    prompt = PROMPT_TEMPLATE.format(
        context=context_text,
        title=video_info['title'],
        description=video_info['shortDescription'],
        channel_name=video_info['channel']['name'],
        channel_url=video_info['channel']['url'],
        channel_subscribers=video_info['channel']['subscribers']['pretty'],
        duration=video_info['duration']['lengthSec'],
        views=video_info['views']['pretty'],
        question=request.query
    )
    response = llm.invoke(prompt)
    return JSONResponse({"content":response.content}, media_type="application/json")

# Test endpoint to check if the server is running
@app.get("/test")
async def read_root():
    return {"message": "Hello World (powered by FastAPI) 🚀"}

# Main entry point to run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)