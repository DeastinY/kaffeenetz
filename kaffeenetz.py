import json
from tqdm import tqdm
import requests
from lxml import html
from pathlib import Path

OUTFILE = Path("ich_trinke_gerade.json")

def parse_page(url):
    response = requests.get(url)
    parsed_page = html.fromstring(response.content)
    return parsed_page


def get_posts(parsed_page):
    return [p.text_content() for p in parsed_page.xpath('//article//blockquote')]


def crawl_posts():
    base_url = "https://www.kaffee-netz.de/threads/ich-trinke-gerade-diesen-espresso.19308/"
    all_urls = [base_url] + [base_url+f"page-{i}" for i in range(1, 302)]  # last page
    with OUTFILE.open("w") as fout:
        for i, url in enumerate(tqdm(all_urls)):
            parsed_page = parse_page(url)
            posts = get_posts(parsed_page)
            fout.write(json.dumps(posts))
            if i % 50 == 0:
                fout.flush()


if __name__ == '__main__':
    crawl_posts()