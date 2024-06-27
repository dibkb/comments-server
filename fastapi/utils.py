import re
def is_valid_youtube_id(video_id):
    regex = r'^[a-zA-Z0-9_-]{11}$'
    return bool(re.match(regex, video_id))

