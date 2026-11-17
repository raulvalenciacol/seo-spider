"""Crawler core shared by the Streamlit UI (app.py) and the MCP server
(mcp_server.py). No Streamlit imports here."""
import copy
import hashlib
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

SPIDER_USER_AGENT = "SEOSpider/1.0 (+https://github.com/raulvalenciacol/seo-spider)"

BROWSER_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0",
]

def build_session_headers(use_spider_ua=False):
    """Pick ONE consistent set of headers for the entire crawl session."""
    if use_spider_ua:
        return {
            "User-Agent": SPIDER_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
    ua = random.choice(BROWSER_USER_AGENTS)
    headers = {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }
    if "Chrome" in ua:
        ver = re.search(r"Chrome/(\d+)", ua).group(1)
        headers["Sec-Ch-Ua"] = f'"Chromium";v="{ver}", "Google Chrome";v="{ver}", "Not-A.Brand";v="8"'
        headers["Sec-Ch-Ua-Mobile"] = "?0"
        headers["Sec-Ch-Ua-Platform"] = '"Windows"' if "Windows" in ua else '"macOS"' if "Mac" in ua else '"Linux"'
    return headers


def get_domain(url):
    return urlparse(url).netloc


def normalize_url(url):
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    if parsed.query:
        return f"{parsed.scheme}://{parsed.netloc}{path}?{parsed.query}"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def get_sitemap_urls(base_url, session):
    urls = set()
    sitemap_locations = [f"{base_url}/sitemap.xml", f"{base_url}/sitemap_index.xml"]
    try:
        r = session.get(f"{base_url}/robots.txt", timeout=10)
        if r.status_code == 200:
            for line in r.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sitemap_locations.append(line.split(":", 1)[1].strip())
    except Exception:
        pass

    def parse_sitemap(url, depth=0):
        if depth > 3:
            return
        try:
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                return
            root = ET.fromstring(r.content)
            ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            for sitemap in root.findall(".//s:sitemap/s:loc", ns):
                parse_sitemap(sitemap.text.strip(), depth + 1)
            for loc in root.findall(".//s:url/s:loc", ns):
                urls.add(loc.text.strip())
        except Exception:
            pass

    for loc in set(sitemap_locations):
        parse_sitemap(loc)
    return urls


def calculate_flesch(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()
    if not sentences or not words:
        return 0, 0, 0
    syllable_count = 0
    for word in words:
        word = word.lower().strip(".,!?;:'\"")
        if not word:
            continue
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count <= 0:
            count = 1
        syllable_count += count
    avg_wps = len(words) / len(sentences)
    avg_syl = syllable_count / len(words)
    flesch = 206.835 - 1.015 * avg_wps - 84.6 * avg_syl
    return round(flesch, 3), len(sentences), round(avg_wps, 3)


def flesch_label(score):
    if score >= 90: return "Very Easy"
    if score >= 80: return "Easy"
    if score >= 70: return "Fairly Easy"
    if score >= 60: return "Standard"
    if score >= 50: return "Fairly Difficult"
    if score >= 30: return "Difficult"
    return "Very Confusing"


def extract_text_content(soup):
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r'\s+', ' ', text)


def get_indexability(meta_robots, canonical, url):
    noindex, reason = False, ""
    if meta_robots and "noindex" in meta_robots.lower():
        noindex, reason = True, "noindex"
    if canonical and normalize_url(canonical) != normalize_url(url):
        noindex, reason = True, "Canonicalised"
    return ("Non-Indexable", reason) if noindex else ("Indexable", "")


def crawl_url(url, session, base_domain, crawl_depth=0):
    row = {
        "Address": url, "Content Type": "", "Status Code": "", "Status": "",
        "Indexability": "", "Indexability Status": "",
        "Title 1": "", "Title 1 Length": 0,
        "Meta Description 1": "", "Meta Description 1 Length": 0,
        "Meta Description 2": "", "Meta Description 2 Length": 0,
        "Meta Keywords 1": "", "Meta Keywords 1 Length": 0,
        "H1-1": "", "H1-1 Length": 0, "H1-2": "", "H1-2 Length": 0,
        "H2-1": "", "H2-1 Length": 0, "H2-2": "", "H2-2 Length": 0,
        "Meta Robots 1": "", "X-Robots-Tag 1": "", "Meta Refresh 1": "",
        "Canonical Link Element 1": "", 'rel="next" 1': "", 'rel="prev" 1': "",
        "Size (bytes)": 0, "Word Count": 0, "Sentence Count": 0,
        "Average Words Per Sentence": 0, "Flesch Reading Ease Score": 0,
        "Readability": "", "Text Ratio": 0,
        "Crawl Depth": crawl_depth,
        "Folder Depth": url.rstrip("/").count("/") - 2,  # subtract 2 for scheme slashes (https://)
        "Response Time": 0, "Redirect URL": "", "Redirect Type": "",
        "Language": "", "Last Modified": "",
        "Hash": "", "URL Encoded Address": url,
        "Crawl Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    discovered_urls = []
    images_found = []

    try:
        start = time.time()
        r = session.get(url, timeout=30, allow_redirects=True)
        # Retry once with backoff if blocked
        if r.status_code in (403, 429):
            time.sleep(random.uniform(2.0, 5.0))
            r = session.get(url, timeout=30, allow_redirects=True)
        elapsed = round(time.time() - start, 3)

        row["Response Time"] = elapsed
        row["Status Code"] = r.status_code
        row["Content Type"] = r.headers.get("Content-Type", "")
        row["Size (bytes)"] = len(r.content)
        row["Last Modified"] = r.headers.get("Last-Modified", "")

        status_map = {
            200: "OK", 301: "Moved Permanently", 302: "Found",
            304: "Not Modified", 307: "Temporary Redirect",
            400: "Bad Request", 403: "Forbidden", 404: "Not Found",
            410: "Gone", 500: "Internal Server Error", 503: "Service Unavailable"
        }
        row["Status"] = status_map.get(r.status_code, str(r.status_code))

        if r.history:
            row["Redirect URL"] = r.url
            row["Redirect Type"] = str(r.history[0].status_code)

        row["X-Robots-Tag 1"] = r.headers.get("X-Robots-Tag", "")

        content_type = r.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            row["Indexability"] = "Non-Indexable"
            row["Indexability Status"] = "Non-HTML"
            return row, discovered_urls, images_found, []

        soup = BeautifulSoup(r.text, "html.parser")

        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            row["Title 1"] = title_tag.string.strip()
            row["Title 1 Length"] = len(row["Title 1"])

        meta_descs = soup.find_all("meta", attrs={"name": re.compile(r"description", re.I)})
        for i, md in enumerate(meta_descs[:2]):
            content = md.get("content", "").strip()
            row[f"Meta Description {i+1}"] = content
            row[f"Meta Description {i+1} Length"] = len(content)

        meta_kw = soup.find("meta", attrs={"name": re.compile(r"keywords", re.I)})
        if meta_kw:
            row["Meta Keywords 1"] = meta_kw.get("content", "").strip()
            row["Meta Keywords 1 Length"] = len(row["Meta Keywords 1"])

        for tag_name, prefix in [("h1", "H1"), ("h2", "H2")]:
            tags = soup.find_all(tag_name)
            for i, tag in enumerate(tags[:2]):
                text = tag.get_text(strip=True)
                row[f"{prefix}-{i+1}"] = text
                row[f"{prefix}-{i+1} Length"] = len(text)

        meta_robots = ""
        mrt = soup.find("meta", attrs={"name": re.compile(r"robots", re.I)})
        if mrt:
            meta_robots = mrt.get("content", "").strip()
            row["Meta Robots 1"] = meta_robots

        mr = soup.find("meta", attrs={"http-equiv": re.compile(r"refresh", re.I)})
        if mr:
            row["Meta Refresh 1"] = mr.get("content", "")

        canonical = ""
        ct = soup.find("link", rel="canonical")
        if ct:
            canonical = ct.get("href", "").strip()
            row["Canonical Link Element 1"] = canonical

        for rt in ["next", "prev"]:
            tag = soup.find("link", rel=rt)
            if tag:
                row[f'rel="{rt}" 1'] = tag.get("href", "")

        html_tag = soup.find("html")
        if html_tag:
            row["Language"] = html_tag.get("lang", "")

        idx, idx_status = get_indexability(meta_robots, canonical, url)
        row["Indexability"] = idx
        row["Indexability Status"] = idx_status

        text_content = extract_text_content(copy.copy(soup))
        row["Word Count"] = len(text_content.split())
        html_size = len(r.text)
        text_size = len(text_content)
        row["Text Ratio"] = round((text_size / html_size * 100) if html_size > 0 else 0, 3)
        flesch, sc, aw = calculate_flesch(text_content)
        row["Flesch Reading Ease Score"] = flesch
        row["Sentence Count"] = sc
        row["Average Words Per Sentence"] = aw
        row["Readability"] = flesch_label(flesch)

        row["Hash"] = hashlib.md5(r.content).hexdigest()

        external_urls = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue
            full_url = urljoin(url, href)
            full_url = normalize_url(full_url)
            if get_domain(full_url) == base_domain:
                discovered_urls.append(full_url)
            else:
                external_urls.append(full_url)

        for img in soup.find_all("img"):
            src = img.get("src", "").strip() or img.get("data-src", "").strip()
            if not src:
                continue
            alt = img.get("alt", None)
            images_found.append({
                "Page URL": url,
                "Image URL": urljoin(url, src),
                "Alt Text": alt if alt is not None else "",
                "Alt Text Status": "Missing" if alt is None else ("Empty" if alt.strip() == "" else "Present"),
                "Alt Text Length": len(alt) if alt else 0,
            })

    except requests.Timeout:
        row["Status Code"] = 0
        row["Status"] = "Timeout"
        external_urls = []
    except requests.ConnectionError:
        row["Status Code"] = 0
        row["Status"] = "Connection Error"
        external_urls = []
    except Exception as e:
        row["Status Code"] = 0
        row["Status"] = f"Error: {str(e)[:50]}"
        external_urls = []

    return row, discovered_urls, images_found, external_urls


def _report(progress, crawled, total, message):
    if progress:
        try:
            progress(crawled, total, message)
        except Exception:
            pass


def parse_header_lines(text):
    """'Name: value' lines -> dict. Blank lines and lines without ':' are ignored."""
    headers = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        if name.strip():
            headers[name.strip()] = value.strip()
    return headers


def run_spider(start_url, max_pages, threads, delay, url_filter="all", proxy_url=None, use_spider_ua=False,
               extra_headers=None, progress=None, should_stop=None):
    """Crawl a site. `progress(crawled, total, message)` is called as pages
    finish; `should_stop()` returning True ends the crawl early. Returns
    (rows, images, blocked_count)."""
    base_domain = get_domain(start_url)
    base_url = f"{urlparse(start_url).scheme}://{base_domain}"

    filter_patterns = {
        "all": None, "products": ["/products/"],
        "collections": ["/collections/"],
        "products_and_collections": ["/products/", "/collections/"],
    }
    active_filter = filter_patterns.get(url_filter)

    def url_matches_filter(url):
        return active_filter is None or any(p in url for p in active_filter)

    session = requests.Session()
    session.headers.update(build_session_headers(use_spider_ua))
    # e.g. Shopify "Crawler access" signature headers, sent with every request.
    if extra_headers:
        session.headers.update(extra_headers)
    if proxy_url:
        session.proxies = {"http": proxy_url, "https": proxy_url}

    # Warm up session — visit homepage first to get cookies (like a real browser)
    _report(progress, 0, max_pages, "Initializing session...")
    try:
        # First visit uses Sec-Fetch-Site: none (direct navigation)
        warmup_headers = {"Sec-Fetch-Site": "none", "Referer": base_url + "/"}
        session.get(base_url, timeout=15, headers=warmup_headers)
        time.sleep(random.uniform(1.5, 3.0))
    except Exception:
        pass
    # All subsequent requests look like same-origin navigation with a referer
    session.headers.update({"Referer": base_url + "/"})

    _report(progress, 0, max_pages, "Fetching sitemap URLs...")
    sitemap_urls = get_sitemap_urls(base_url, session)
    _report(progress, 0, max_pages, f"Found {len(sitemap_urls)} URLs in sitemap")

    queue = [(start_url, 0)]
    for su in sitemap_urls:
        if get_domain(su) == base_domain:
            queue.append((su, 1))

    visited = set()
    results = []
    all_images = []
    blocked_count = 0
    inlinks = defaultdict(set)
    outlinks = defaultdict(lambda: {"internal": set(), "external": set()})
    crawled = 0
    lock = threading.Lock()

    def process_url(url, depth):
        nonlocal crawled, blocked_count
        norm = normalize_url(url)
        with lock:
            if norm in visited:
                return []
            visited.add(norm)

        row, discovered, images, external = crawl_url(url, session, base_domain, depth)

        with lock:
            if url_matches_filter(url):
                results.append(row)
                all_images.extend(images)
                for d_url in discovered:
                    inlinks[d_url].add(url)
                    outlinks[url]["internal"].add(d_url)
                for e_url in external:
                    outlinks[url]["external"].add(e_url)
            if row.get("Status Code") in (403, 429, 503):
                blocked_count += 1
            crawled += 1
        blocked_warning = f" | {blocked_count} blocked" if blocked_count > 0 else ""
        _report(progress, crawled, max_pages, f"Crawled {crawled} / {max_pages}{blocked_warning} — {url[:70]}")
        time.sleep(delay + random.uniform(0, delay * 0.5))
        return [(u, depth + 1) for u in discovered if get_domain(u) == base_domain]

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {}
        queue_idx = 0

        while queue_idx < len(queue) and queue_idx < threads * 2:
            url, depth = queue[queue_idx]
            norm = normalize_url(url)
            if norm not in visited:
                futures[executor.submit(process_url, url, depth)] = True
            queue_idx += 1

        while futures and crawled < max_pages and not (should_stop and should_stop()):
            completed, _ = wait(set(futures.keys()), return_when=FIRST_COMPLETED)
            for f in completed:
                try:
                    new_urls = f.result()
                    for new_url, new_depth in new_urls:
                        norm = normalize_url(new_url)
                        with lock:
                            if norm not in visited and crawled < max_pages:
                                queue.append((new_url, new_depth))
                except Exception:
                    pass
                del futures[f]

            while queue_idx < len(queue) and len(futures) < threads * 2 and crawled < max_pages:
                url, depth = queue[queue_idx]
                norm = normalize_url(url)
                with lock:
                    if norm not in visited:
                        futures[executor.submit(process_url, url, depth)] = True
                queue_idx += 1

    total = len(results)
    for row in results:
        url = row["Address"]
        row["Inlinks"] = len(inlinks.get(url, set()))
        row["Unique Inlinks"] = row["Inlinks"]
        row["Outlinks"] = len(outlinks[url]["internal"]) + len(outlinks[url]["external"])
        row["Unique Outlinks"] = row["Outlinks"]
        row["External Outlinks"] = len(outlinks[url]["external"])
        row["Unique External Outlinks"] = row["External Outlinks"]
        row["% of Total"] = round(row["Inlinks"] / total * 100, 3) if total > 0 else 0

    return results, all_images, blocked_count


