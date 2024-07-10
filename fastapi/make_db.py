from youtube_transcript_api import YouTubeTranscriptApi
from langchain.docstore.document import Document
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import requests

load_dotenv()

def get_description(videoid):
    video_info = {}
    res = requests.get(f"https://comment-sense-nodeapi.vercel.app/get-info/{videoid}")
    data = res.json()
    video_info['title'] = data['title']
    video_info['shortDescription'] = data['shortDescription']
    video_info['channel'] = data['channel']
    video_info['duration'] = data['duration']
    video_info['views'] = data['views']
    return video_info

def make_vector_database(id):
    CHROMA_PATH = f'data/chroma_db_{id}'
    if os.path.exists(CHROMA_PATH):
        print("Data base already exists...")
        return
    generate_data_store(id)

def generate_data_store(id):
    documents = transcript_loader(id)
    chunks = split_text(documents)
    save_to_chroma(chunks, id)

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks, id):
    CHROMA_PATH = f'data/chroma_db_{id}'
    if os.path.exists(CHROMA_PATH):
        return
    Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def transcript_loader(videoid):
    trans=[]
    page_content=""
    transcript_list = YouTubeTranscriptApi.list_transcripts(videoid)
    for transcript in transcript_list:
        # translating the transcript will return another transcript object
        trans = transcript.translate('en').fetch()
        for t in trans:
            if(t['text'] and t['text'].strip() != ""):
                page_content+=(" "+t['text'])
    return [Document(
        page_content,
        metadata={
            "source":videoid,
        }
    )]
