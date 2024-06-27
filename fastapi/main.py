from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


from downloader import YoutubeCommentDownloader
from services import process_comments, async_generator
from utils import is_valid_youtube_id

app = FastAPI()

origins = [
    "http://localhost",
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

@app.get("/video")
async def get_video(ytid: str,start: int,end: int,sort:int = Query(...)):
    # Handle Youtubeid
    if(is_valid_youtube_id(ytid) == False):
        raise HTTPException(status_code=404, detail="Invalid YouTube videoid")

    # Handle sort parameters
    if sort != 0 and sort != 1:
        raise HTTPException(status_code=404, detail="Invalid sort value. Expected either 0 or 1")
    comments = async_generator(downloader.get_comments(ytid, sort_by=sort))
    processed = process_comments(comments, start, end)
    
    res = []
    async for comment in processed:
        res.append(comment)
    
    return JSONResponse(res, media_type="application/json")



@app.get("/test")
async def read_root():
    return {"message": "Hello World (powered by FastAPI) 🚀"}

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
