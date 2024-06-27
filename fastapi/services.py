import asyncio
from translate import GoogleTranslator
from langdetect import detect, LangDetectException
from models import sentiment_pipeline,emotion_pipeline

translator = GoogleTranslator()

async def get_translated(text):
    loop = asyncio.get_event_loop()
    translation = await loop.run_in_executor(None, translator.translate, text, "", "en")
    return ''.join([t['trans'] for t in translation['sentences']])

async def process_comment(comment):
    text = comment["text"]
    try:
        if detect(text) != 'en':
            comment['translated'] = await get_translated(text)
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


async def process_comments(comments, start: int, end: int):
    current_index = 0
    tasks = []

    async for comment in comments:
        if current_index < start:
            continue
        if current_index >= end:
            break

        task = asyncio.create_task(process_comment(comment))
        tasks.append(task)
        current_index += 1

    for task in asyncio.as_completed(tasks):
        yield await task


async def async_generator(sync_generator):
    for item in sync_generator:
        yield item
