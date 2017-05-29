import math
import json
from tqdm import tqdm
import logging
import requests
from lxml import html
from pathlib import Path
from textblob import TextBlob as tb
from nltk.tag.stanford import StanfordNERTagger, StanfordPOSTagger
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO)
FILE_DATA = Path("ich_trinke_gerade.json")
FILE_PROCESSED = Path("ich_trinke_gerade_processed.json")
FILE_TFIDF = Path("tfidf.json")

def parse_page(url):
    response = requests.get(url)
    parsed_page = html.fromstring(response.content)
    return parsed_page


def get_posts(parsed_page):
    return [p.text_content() for p in parsed_page.xpath('//article//blockquote')]


def crawl_posts():
    base_url = "https://www.kaffee-netz.de/threads/ich-trinke-gerade-diesen-espresso.19308/"
    all_urls = [base_url] + [base_url+f"page-{i}" for i in range(1, 302)]  # last page
    post_data = []
    for url in tqdm(all_urls):
        parsed_page = parse_page(url)
        posts = get_posts(parsed_page)
        post_data.append(posts)
    FILE_DATA.write_text(json.dumps(post_data, indent=4))


def load_posts():
    logging.info("Loading post data from file")
    posts =  json.loads(FILE_DATA.read_text(encoding="utf-8"))
    logging.info(f"Loaded {sum([len(p) for p in posts])} posts")
    return posts


def load_processed():
    logging.info("Loading processed data")
    posts = json.loads(FILE_PROCESSED.read_text(encoding="utf-8"))
    logging.info(f"Loaded {sum([len(p) for p in posts])} posts")
    return posts


def load_tfidf():
    logging.info("Loading TFIDF")
    tfidf = json.loads(FILE_TFIDF.read_text(encoding="utf-8"))
    logging.info(f"Loaded {len(tfidf)} tfidf scores")
    return tfidf


def calculate_tf(word, blob):
    return blob.words.count(word) / len(blob.words)


def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)


def calculate_idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))


def calculate_tfidf(word, blob, bloblist):
    return calculate_tf(word, blob) * calculate_idf(word, bloblist)


def get_tfidf_scores(posts):
    scores = {}
    bloblist = [tb(post['text']) for post in posts]
    wordlist = [[ne[0] for ne in post['named_entities']] for post in posts]
    for i, blob in enumerate(tqdm(bloblist)):
        for word in wordlist[i]:
            scores[word] = calculate_tfidf(word, blob, bloblist)
    FILE_TFIDF.write_text(json.dumps(scores, indent=4))


def preprocess(raw_data):
    """Input is a list of threads containing posts as strings. Output a list of single preprocessed posts"""
    logging.info("Preprocessing")
    posts = []
    i = 0
    for thread in tqdm(raw_data):
        for post in thread:
            text = " ".join(post.split())  # Join multi line post to single line string
            posts.append({
                "id": i,
                "text": text
            })
            i += 1
    return posts


def tokenize(posts):
    logging.info("Tokenizing")
    for post in tqdm(posts):
        post["tokens"] = [p for p in word_tokenize(post["text"]) if not p in stopwords.words('german')]
    return posts


def pos_tag(posts):
    logging.info("Extracting POS")
    #pos_tagger = StanfordPOSTagger(model_filename="/home/ric/stanford/models/german-ud.tagger")
    #post = pos_tagger.tag(posts)
    #post = [p for p in post if p[1] not in ["PUNCT"]]
    #post = ne_chunk(post)
    return posts


def extract_ner(posts):
    logging.info("Extracting NER")
    ner_tagger = StanfordNERTagger(model_filename="/home/ric/stanford/models/ner/german.conll.hgc_175m_600.crf.ser.gz")
    all_ne = set()
    for p in tqdm(posts):
        tagged = ner_tagger.tag(p["tokens"])
        named_entities = [t for t in tagged if t[1] !='O']
        p['tagged'] = tagged
        p['named_entities'] = named_entities
        for n in named_entities:
            all_ne.add(n[1])
    return posts, list(all_ne)


def generate_processed(crawl=False):
    if crawl:
        crawl_posts()
    raw_data = load_posts()
    posts = preprocess(raw_data)
    posts = tokenize(posts)
    posts = keyword_check(posts)
    posts, all_ne = extract_ner(posts)
    FILE_PROCESSED.write_text(json.dumps(posts, indent=4))


def get_unique_ne(posts):
    all_ne = [[p[0] for p in post['named_entities']] for post in posts]
    all_unique = set()
    for ne in all_ne:
        for n in ne:
            all_unique.add(n)
    return all_unique


def filter_ne(posts, tfidf):
    logging.info("Filtering named entities")
    filter_words = ['klicke']
    for post in tqdm(posts):
        filtered = []
        for ne in post["named_entities"]:
            ne = ne[0].lower()
            if any([fw in ne for fw in filter_words]):
                continue
            else:
                filtered.append(ne)
        post['filtered_ne'] = filtered
    return posts


def keyword_check(posts):
    keywords = ["Name", "Bohne", "Gramm", "Gemahlen" ,"Bezogen", "Gekauft", "Eigenschaften", "Wieder", "kaufen"]
    for post in posts:
        hits = [key in post["text"] for key in keywords].count(True) / len(keywords)
        post["keyword_score"] = hits
    return posts


def get_coffee_posts(posts, threshold=0.9):
    coffee_posts = []
    for post in posts:
        if post["keyword_score"] > threshold:
            coffee_posts.append({
                "Name": "",
                "Keywords:": post["filtered_ne"],
                "Full Post": post["text"]
            })
    return coffee_posts


if __name__ == '__main__':
    posts = load_processed()
    coffee_posts = get_coffee_posts(posts)
    print(json.dumps(coffee_posts, indent=4))
    print(len(coffee_posts))

