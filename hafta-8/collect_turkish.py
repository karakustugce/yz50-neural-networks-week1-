"""Vikikaynak API'sinden klasik Türkçe metin ve kaynak listesi toplar."""
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://tr.wikisource.org/w/api.php"
ROOT = Path(__file__).resolve().parent
CATEGORIES = [
    "Ömer Seyfettin hikayeleri",
    "Mehmet Âkif Ersoy şiirleri",
    "Namık Kemal şiirleri",
    "Tevfik Fikret şiirleri",
    "Ziya Gökalp Şiirleri",
    "Türkiye'de yayımlanan kamu malı eserler",
]
AGENT = "YZ50-week8-character-LM/1.0 educational"


def query(params):
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": AGENT})
    with urllib.request.urlopen(req, timeout=45) as res:
        return json.loads(res.read().decode("utf-8"))


def category_members(category):
    args = {"action": "query", "list": "categorymembers",
            "cmtitle": "Category:" + category, "cmnamespace": 0,
            "cmlimit": "max", "format": "json"}
    names = []
    while True:
        obj = query(args)
        names += [p["title"] for p in obj["query"]["categorymembers"]]
        if "continue" not in obj:
            return names
        args.update(obj["continue"])


def main():
    titles = set()
    for category in CATEGORIES:
        try:
            titles.update(category_members(category))
        except Exception as e:
            print(f"Kategori atlandı: {category}: {e}")
    pages = []
    names = sorted(titles)
    for i in range(0, len(names), 20):
        data = query({"action": "query", "prop": "extracts|info", "explaintext": 1,
                      "inprop": "url", "redirects": 1, "titles": "|".join(names[i:i+20]),
                      "format": "json"})
        pages.extend(data["query"]["pages"].values())
        time.sleep(0.15)
    corpus, sources = [], []
    for page in sorted(pages, key=lambda p: p["title"]):
        body = (page.get("extract") or "").strip()
        if len(body) < 300:
            continue
        corpus.append(f"\n\n### {page['title']}\n\n{body}")
        sources.append({"title": page["title"], "url": page.get("fullurl", ""),
                        "page_id": page.get("pageid", ""), "characters": len(body)})
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    text = "".join(corpus).strip() + "\n"
    (data_dir / "turkish.txt").write_text(text, encoding="utf-8")
    with (data_dir / "turkish_sources.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "url", "page_id", "characters"])
        writer.writeheader()
        writer.writerows(sources)
    meta = {"categories": CATEGORIES, "page_count": len(sources),
            "characters_including_titles": len(text),
            "unique_unicode_characters": len(set(text)),
            "source": "Turkish Wikisource / Vikikaynak",
            "attribution": "Use turkish_sources.csv for attribution and inspect the license shown on each source page."}
    (data_dir / "turkish_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("Yaklaşık 1 MB hedefi için karakter sayısını kontrol edin; gerekirse README'deki şekilde kategori ekleyin.")


if __name__ == "__main__":
    main()
