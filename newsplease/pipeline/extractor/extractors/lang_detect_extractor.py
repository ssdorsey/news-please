import locale
import re
import os
from pathlib import Path

from lxml import html
import urllib.request

import fasttext
import warnings
warnings.filterwarnings("ignore")

from .abstract_extractor import AbstractExtractor

_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).as_posix()

def get_models_path(path):
    return Path('/'.join(_ROOT.split('/')[:-1]), 'models', path).joinpath()

# if the models folder doesn't exist, create it
if not os.path.exists(get_models_path('')):
    # os.makedirs(get_models_path(''))
    Path(get_models_path('')).mkdir(parents=True)

# download the fasttext model if it's not available
if 'lid.176.bin' not in os.listdir(get_models_path('')):
    url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
    urllib.request.urlretrieve(url, get_models_path('lid.176.bin'))

lang_detection = fasttext.load_model(str(get_models_path('lid.176.bin'))) 

class LangExtractor(AbstractExtractor):
    """This class implements LangDetect as an article extractor but it can only
    detect the extracted language (en, de, ...).

    """

    def __init__(self):
        self.name = "langdetect"
        self.langcode_pattern = re.compile(r'\b[a-zA-Z]{2}(?=([-_]|\b))')

    def _language(self, item):
        """Returns the language of the extracted article by analyzing metatags and inspecting the visible text
        with langdetect"""

        response = item['spider_response'].body
        root = html.fromstring(response)

        lang = None

        # Look for <article> elements and inspect the one with the largest payload with langdetect
        article_list = []
        for article in root.xpath('//article'):
            article_list.append(re.sub(r'\s+', ' ', article.text_content().strip()))
            if len(article_list) > 0:
                lang = lang_detection.predict(max(article_list))
                if lang:
                    lang = lang[0][0][-2:]
                    return lang

        # Analyze the whole body with langdetect
        if lang is None:
            try:
                lang = lang_detection.predict(root.text_content().strip())[0][0][-2:]
            except:
                pass

        if lang:
            return lang

        # Check for lang-attributes
        lang = root.get('lang')

        if lang is None:
            lang = root.get('xml:lang')

        # Check for general meta tags
        if lang is None:
            meta = root.cssselect('meta[name="language"]')
            if len(meta) > 0:
                lang = meta[0].get('content')

        # Check for open graph tags
        if lang is None:
            meta = root.cssselect('meta[property="og:locale"]')
            if len(meta) > 0:
                lang = meta[0].get('content')

        # Try to normalize output
        if lang is not None:
            # First search for suitable locale in the original output
            matches = self.langcode_pattern.search(lang)
            if matches is not None:
                lang = matches.group(0)
            else:
                # If no match was found, normalize the original output and search again
                normalized = locale.normalize(re.split(r'\s|;|,', lang.strip())[0])
                matches = self.langcode_pattern.search(normalized)
                if matches is not None:
                    lang = matches.group(0)

        return lang
