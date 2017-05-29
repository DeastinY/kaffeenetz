import re
import json
import logging
from pathlib import Path

import requests
from lxml import html
from nltk.corpus import stopwords
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import word_tokenize
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
FILE_DATA = Path("ich_trinke_gerade.json")
FILE_PROCESSED = Path("ich_trinke_gerade_processed.json")
FILE_FINAL = Path("results.json")


def parse_page(url):
    response = requests.get(url)
    parsed_page = html.fromstring(response.content)
    return parsed_page


def get_posts(parsed_page):
    return [p.text_content() for p in parsed_page.xpath('//article//blockquote')]


def crawl_posts():
    base_url = "https://www.kaffee-netz.de/threads/ich-trinke-gerade-diesen-espresso.19308/"
    all_urls = [base_url] + [base_url + f"page-{i}" for i in range(1, 302)]  # last page
    post_data = []
    for url in tqdm(all_urls):
        parsed_page = parse_page(url)
        posts = get_posts(parsed_page)
        post_data.append(posts)
    FILE_DATA.write_text(json.dumps(post_data, indent=4, ensure_ascii=False))


def load_posts():
    logging.info("Loading post data from file")
    posts = json.loads(FILE_DATA.read_text().encode('utf-8', 'ignore'))
    logging.info(f"Loaded {sum([len(p) for p in posts])} posts")
    return posts


def load_processed():
    logging.info("Loading processed data")
    posts = json.loads(FILE_PROCESSED.read_text().encode('utf-8', 'ignore'))
    logging.info(f"Loaded {sum([len(p) for p in posts])} posts")
    return posts


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
        post["tokens"] = [p for p in word_tokenize(post["text"]) if p not in stopwords.words('german')]
    return posts


def pos_tag(posts):
    logging.info("Extracting POS")
    # pos_tagger = StanfordPOSTagger(model_filename="/home/ric/stanford/models/german-ud.tagger")
    # post = pos_tagger.tag(posts)
    # post = [p for p in post if p[1] not in ["PUNCT"]]
    # post = ne_chunk(post)
    return posts


def extract_ner(posts):
    logging.info("Extracting NER")
    ner_tagger = StanfordNERTagger(model_filename="/home/ric/stanford/models/ner/german.conll.hgc_175m_600.crf.ser.gz")
    all_ne = set()
    for p in tqdm(posts):
        tagged = ner_tagger.tag(p["tokens"])
        named_entities = [t for t in tagged if t[1] != 'O']
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
    FILE_PROCESSED.write_text(json.dumps(posts, indent=4, ensure_ascii=False))


def get_unique_ne(posts):
    all_ne = [[p[0] for p in post['named_entities']] for post in posts]
    all_unique = set()
    for ne in all_ne:
        for n in ne:
            all_unique.add(n)
    return all_unique


def filter_ne(posts):
    logging.info("Filtering named entities")
    filter_words = ['klicke', "bohne", "espresso", "kaffee", "mhd", "packungsgröße", ]
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
    keywords = ["Name", "Bohne", "Gramm", "Gemahlen", "Bezogen", "Gekauft", "Eigenschaften", "Wieder", "kaufen"]
    for post in posts:
        hits = [key in post["text"] for key in keywords].count(True) / len(keywords)
        post["keyword_score"] = hits
    return posts


def get_coffee_posts(posts, threshold=0.9):
    logging.info("Grouping coffee posts")
    coffee_posts = []
    for post in posts:
        if post["keyword_score"] > threshold:
            coffee_posts.append({
                "Full Post": post["text"]
            })
    return coffee_posts


def extract_coffee_details(coffee_posts):
    logging.info("Extracting Details")
    keywords = ["Name der Bohne", "Packungsgröße", "Gemahlen und Bezogen mit",
                "Gekauft am", "MHD", "Eigenschaften", "Geschmack", "Wieder kaufen"]
    keywords_colon = [kw+":" for kw in keywords]
    re_name = re.compile(r"(?<=Name der Bohne: ).+?(?=.?("+r"|".join(keywords_colon)+r"))", re.IGNORECASE)
    re_packung = re.compile(r"(?<=Packungsgröße: ).+?(?=.?("+r"|".join(keywords_colon)+r"))", re.IGNORECASE)
    re_gerät = re.compile(r"(?<=Gemahlen und Bezogen mit: ).+?(?=.?("+r"|".join(keywords_colon)+r"))", re.IGNORECASE)
    re_gekauft = re.compile(r"(?<=Gekauft am: ).+?(?=.?("+r"|".join(keywords_colon)+r"))", re.IGNORECASE)
    re_mhd = re.compile(r"(?<=MHD: ).+?(?=.?("+r"|".join(keywords_colon)+r"))", re.IGNORECASE)
    re_eigenschaften = re.compile(r"(?<=Eigenschaften: ).+?(?=.?("+r"|".join(keywords_colon)+r"))", re.IGNORECASE)
    re_geschmack = re.compile(r"(?<=Geschmack: ).+?(?=.?("+r"|".join(keywords_colon)+r"))", re.IGNORECASE)
    re_wiederkaufen = re.compile(r"(?<=Wieder kaufen: ).+", re.IGNORECASE)
    all_re = {
        "Name": re_name,
        "Packungsgröße": re_packung,
        "Maschine": re_gerät,
        "Gekauft am": re_gekauft,
        "MHD": re_mhd,
        "Eigenschaften": re_eigenschaften,
        "Geschmack": re_geschmack,
        "Wiederkaufen": re_wiederkaufen
    }

    errorcnt = 0
    for cp in tqdm(coffee_posts):
        text = cp["Full Post"]
        for name, regex in all_re.items():
            result = regex.search(text)
            if result:
                value = result.group()
                for kw in keywords:
                    if __name__ == '__main__':
                        value = value.replace(kw, '')
                cp[name] = value.strip()
            else:
                errorcnt += 1
                cp[name] = ""
                logging.debug(f"Could not extract {name} from a post. Here it is : \n{text}")
    logging.info(f"Processed {len(coffee_posts)} posts with {errorcnt} extraction errors.")
    FILE_FINAL.write_text(json.dumps(coffee_posts, indent=4, ensure_ascii=False), encoding='utf-8')
    return coffee_posts


if __name__ == '__main__':
    posts = load_processed()
    coffee_posts = get_coffee_posts(posts, threshold=0.9)
    coffee_posts = extract_coffee_details(coffee_posts)

