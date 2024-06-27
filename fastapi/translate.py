import requests

class GoogleTranslator:
    def __init__(self):
        pass

    def translate(self, text, from_lang, to_lang):
        url = 'https://translate.googleapis.com/translate_a/single'

        params = {
        'client': 'gtx',
        'sl': 'auto',
        'tl': to_lang,
        'hl': from_lang,
        'dt': ['t'],
        'dj': '1',
        'source': 'popup6',
        'q': text
        }

        return requests.get(url, params=params).json()