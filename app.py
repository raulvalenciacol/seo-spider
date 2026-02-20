#!/usr/bin/env python3
"""
SEO Spider - Free Screaming Frog Alternative
Built by Raul @ MaleBasics Corp
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import pandas as pd
import hashlib
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading

# ─── Page Config ───
st.set_page_config(page_title="SEO Spider — Free Screaming Frog Alternative", page_icon="🕷️", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.4rem; font-weight: 700; margin-bottom: 0; }
    .sub-header { font-size: 1.05rem; color: #888; margin-top: 0; margin-bottom: 1rem; }
    .stProgress > div > div > div > div { background-color: #4fc3f7; }
    .donate-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        color: white;
        margin-top: 1rem;
    }
    .donate-box a { color: #ffd700; text-decoration: none; font-weight: bold; }
    .footer { text-align: center; color: #888; font-size: 0.8rem; margin-top: 2rem; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

if "crawl_running" not in st.session_state:
    st.session_state.crawl_running = False
if "crawl_results" not in st.session_state:
    st.session_state.crawl_results = None
if "image_results" not in st.session_state:
    st.session_state.image_results = None

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Sec-Ch-Ua": '"Chromium";v="126", "Google Chrome";v="126", "Not-A.Brand";v="8"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Cache-Control": "max-age=0",
}


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
    except:
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
        except:
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
        "Folder Depth": url.rstrip("/").count("/") - 2,
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
            return row, discovered_urls, images_found

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

        text_content = extract_text_content(BeautifulSoup(r.text, "html.parser"))
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

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue
            full_url = urljoin(url, href)
            full_url = normalize_url(full_url)
            if get_domain(full_url) == base_domain:
                discovered_urls.append(full_url)

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
    except requests.ConnectionError:
        row["Status Code"] = 0
        row["Status"] = "Connection Error"
    except Exception as e:
        row["Status Code"] = 0
        row["Status"] = f"Error: {str(e)[:50]}"

    return row, discovered_urls, images_found


def run_spider(start_url, max_pages, threads, delay, progress_bar, status_text, url_filter="all"):
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
    session.headers.update(HEADERS)

    # Warm up session
    status_text.text("🔐 Initializing session...")
    try:
        session.get(base_url, timeout=15)
        time.sleep(1)
    except:
        pass

    status_text.text("📄 Fetching sitemap URLs...")
    sitemap_urls = get_sitemap_urls(base_url, session)
    status_text.text(f"📄 Found {len(sitemap_urls)} URLs in sitemap")

    queue = [(start_url, 0)]
    for su in sitemap_urls:
        if get_domain(su) == base_domain:
            queue.append((su, 1))

    visited = set()
    results = []
    all_images = []
    inlinks = defaultdict(set)
    outlinks = defaultdict(lambda: {"internal": set(), "external": set()})
    crawled = 0
    lock = threading.Lock()

    def process_url(url, depth):
        nonlocal crawled
        norm = normalize_url(url)
        with lock:
            if norm in visited:
                return []
            visited.add(norm)

        row, discovered, images = crawl_url(url, session, base_domain, depth)

        if url_matches_filter(url):
            with lock:
                results.append(row)
                all_images.extend(images)
                for d_url in discovered:
                    inlinks[d_url].add(url)
                    outlinks[url]["internal"].add(d_url)

        crawled += 1
        progress_bar.progress(min(crawled / max_pages, 1.0))
        status_text.text(f"🕷️ Crawled {crawled} / {max_pages} — {url[:70]}...")
        time.sleep(delay)
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

        while futures and crawled < max_pages:
            done = [f for f in list(futures.keys()) if f.done()]
            if not done:
                time.sleep(0.05)
                continue
            for f in done:
                try:
                    new_urls = f.result()
                    for new_url, new_depth in new_urls:
                        norm = normalize_url(new_url)
                        with lock:
                            if norm not in visited and crawled < max_pages:
                                queue.append((new_url, new_depth))
                except:
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

    return results, all_images


# ─── UI ───

st.markdown('<p class="main-header">🕷️ SEO Spider</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Free Screaming Frog alternative — crawl any site, get full SEO audit</p>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Crawl Settings")
    url_input = st.text_input("Website URL", placeholder="https://yoursite.com", help="Include https://")

    st.markdown("**Page Type Filter**")
    url_filter = st.selectbox("Crawl scope", [
        ("All Pages", "all"),
        ("Products Only (/products/)", "products"),
        ("Collections Only (/collections/)", "collections"),
        ("Products + Collections", "products_and_collections"),
    ], format_func=lambda x: x[0])

    max_pages = st.slider("Max Pages", 10, 2000, 200, step=10,
                          help="Free hosted version limited to 2,000 pages per crawl")
    threads = st.slider("Threads", 1, 5, 3, help="Concurrent requests")
    delay = st.slider("Delay (seconds)", 0.1, 2.0, 0.3, step=0.1)

    st.markdown("---")

    # Donation box
    st.markdown("""
    <div class="donate-box">
        <p style="font-size: 1.1rem; margin-bottom: 8px;">☕ Like this tool?</p>
        <p style="font-size: 0.85rem; margin-bottom: 12px;">Built with ❤️ by a fellow e-commerce operator. If it saves you time (and $$$), consider buying me a coffee!</p>
        <a href="https://ko-fi.com/CHANGETHIS" target="_blank" style="background: #ffd700; color: #333; padding: 8px 20px; border-radius: 6px; display: inline-block;">☕ Buy Me a Coffee</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚠️ Note:** Some sites with heavy bot protection (Shopify stores with Cloudflare) may block the crawler. Works great on WordPress, WooCommerce, Magento, and most other platforms.")

    crawl_button = st.button("🚀 Start Crawl", type="primary", use_container_width=True,
                             disabled=st.session_state.crawl_running)

if crawl_button and url_input:
    url = url_input.strip()
    if not url.startswith("http"):
        url = f"https://{url}"
    st.session_state.crawl_running = True
    st.session_state.crawl_results = None
    st.session_state.image_results = None

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        data, images = run_spider(url, max_pages, threads, delay, progress_bar, status_text, url_filter[1])
        st.session_state.crawl_results = data
        st.session_state.image_results = images
        status_text.text(f"✅ Done! {len(data)} pages crawled. {len(images)} images audited.")
    except Exception as e:
        st.error(f"Crawl failed: {str(e)}")
    finally:
        st.session_state.crawl_running = False

if st.session_state.crawl_results:
    data = st.session_state.crawl_results
    columns = [
        "Address", "Content Type", "Status Code", "Status",
        "Indexability", "Indexability Status",
        "Title 1", "Title 1 Length",
        "Meta Description 1", "Meta Description 1 Length",
        "Meta Description 2", "Meta Description 2 Length",
        "Meta Keywords 1", "Meta Keywords 1 Length",
        "H1-1", "H1-1 Length", "H1-2", "H1-2 Length",
        "H2-1", "H2-1 Length", "H2-2", "H2-2 Length",
        "Meta Robots 1", "X-Robots-Tag 1", "Meta Refresh 1",
        "Canonical Link Element 1", 'rel="next" 1', 'rel="prev" 1',
        "Size (bytes)", "Word Count", "Sentence Count",
        "Average Words Per Sentence", "Flesch Reading Ease Score", "Readability",
        "Text Ratio", "Crawl Depth", "Folder Depth",
        "Inlinks", "Unique Inlinks", "% of Total",
        "Outlinks", "Unique Outlinks",
        "External Outlinks", "Unique External Outlinks",
        "Response Time", "Last Modified", "Redirect URL", "Redirect Type",
        "Language", "Hash", "URL Encoded Address", "Crawl Timestamp",
    ]

    df = pd.DataFrame(data)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    df = df[columns]
    html_df = df[df["Content Type"].str.contains("text/html", na=False)]

    st.markdown("### 📊 Crawl Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total URLs", len(df))
    c2.metric("HTML Pages", len(html_df))
    c3.metric("200 OK", len(df[df["Status Code"] == 200]))
    c4.metric("301/302", len(df[df["Status Code"].isin([301, 302])]))
    c5.metric("404s", len(df[df["Status Code"] == 404]))
    c6.metric("Indexable", len(df[df["Indexability"] == "Indexable"]))

    st.markdown("### 🔍 SEO Issues")
    c1, c2, c3, c4 = st.columns(4)
    missing_title = html_df[html_df["Title 1"].astype(str).str.strip() == ""]
    missing_desc = html_df[html_df["Meta Description 1"].astype(str).str.strip() == ""]
    missing_h1 = html_df[html_df["H1-1"].astype(str).str.strip() == ""]
    dup_titles = html_df[html_df.duplicated(subset=["Title 1"], keep=False) & (html_df["Title 1"].astype(str).str.strip() != "")]
    c1.metric("Missing Title", len(missing_title))
    c2.metric("Missing Meta Desc", len(missing_desc))
    c3.metric("Missing H1", len(missing_h1))
    c4.metric("Duplicate Titles", len(dup_titles))

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📄 All URLs", "🌐 HTML Only", "⚠️ Issues", "🖼️ Images", "🔗 Redirects", "📈 Content"
    ])

    with tab1:
        st.dataframe(df, use_container_width=True, height=500)
    with tab2:
        st.dataframe(html_df, use_container_width=True, height=500)
    with tab3:
        issue_type = st.selectbox("Issue Type", [
            "Missing Title", "Missing Meta Description", "Missing H1",
            "Duplicate Titles", "Non-Indexable Pages", "Long Titles (>60 chars)",
            "Long Meta Descriptions (>160 chars)", "Thin Content (<100 words)"
        ])
        if issue_type == "Missing Title":
            st.dataframe(missing_title[["Address", "Title 1", "H1-1"]], use_container_width=True)
        elif issue_type == "Missing Meta Description":
            st.dataframe(missing_desc[["Address", "Title 1", "Meta Description 1"]], use_container_width=True)
        elif issue_type == "Missing H1":
            st.dataframe(missing_h1[["Address", "Title 1", "H1-1"]], use_container_width=True)
        elif issue_type == "Duplicate Titles":
            st.dataframe(dup_titles[["Address", "Title 1"]].sort_values("Title 1"), use_container_width=True)
        elif issue_type == "Non-Indexable Pages":
            st.dataframe(html_df[html_df["Indexability"] == "Non-Indexable"][["Address", "Indexability Status", "Canonical Link Element 1", "Meta Robots 1"]], use_container_width=True)
        elif issue_type == "Long Titles (>60 chars)":
            st.dataframe(html_df[html_df["Title 1 Length"] > 60][["Address", "Title 1", "Title 1 Length"]], use_container_width=True)
        elif issue_type == "Long Meta Descriptions (>160 chars)":
            st.dataframe(html_df[html_df["Meta Description 1 Length"] > 160][["Address", "Meta Description 1", "Meta Description 1 Length"]], use_container_width=True)
        elif issue_type == "Thin Content (<100 words)":
            st.dataframe(html_df[html_df["Word Count"] < 100][["Address", "Title 1", "Word Count"]], use_container_width=True)

    with tab4:
        if st.session_state.image_results:
            img_df = pd.DataFrame(st.session_state.image_results)
            ic1, ic2, ic3, ic4 = st.columns(4)
            missing_alt = img_df[img_df["Alt Text Status"] == "Missing"]
            empty_alt = img_df[img_df["Alt Text Status"] == "Empty"]
            present_alt = img_df[img_df["Alt Text Status"] == "Present"]
            ic1.metric("Total Images", len(img_df))
            ic2.metric("Missing Alt", len(missing_alt))
            ic3.metric("Empty Alt", len(empty_alt))
            ic4.metric("With Alt Text", len(present_alt))
            img_filter = st.selectbox("Show", ["All Images", "Missing Alt Text", "Empty Alt Text", "Has Alt Text"], key="img_filter")
            show_map = {"Missing Alt Text": missing_alt, "Empty Alt Text": empty_alt, "Has Alt Text": present_alt}
            st.dataframe(show_map.get(img_filter, img_df), use_container_width=True, height=500)
        else:
            st.info("No image data collected.")

    with tab5:
        redirects = df[df["Redirect URL"].astype(str).str.strip() != ""]
        st.dataframe(redirects[["Address", "Status Code", "Redirect URL", "Redirect Type"]], use_container_width=True)

    with tab6:
        st.markdown("**Content Analysis (HTML pages)**")
        content_df = html_df[["Address", "Word Count", "Text Ratio", "Flesch Reading Ease Score", "Readability", "Sentence Count"]].copy()
        st.dataframe(content_df.sort_values("Word Count"), use_container_width=True, height=400)
        if len(html_df) > 0:
            st.markdown("**Word Count Distribution**")
            st.bar_chart(html_df["Word Count"].value_counts().sort_index().head(50))

    st.markdown("---")
    st.markdown("### 💾 Download Reports")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ All URLs (CSV)", df.to_csv(index=False), "crawl_all.csv", "text/csv", use_container_width=True)
    with c2:
        st.download_button("⬇️ HTML Only (CSV)", html_df.to_csv(index=False), "crawl_html.csv", "text/csv", use_container_width=True)
    with c3:
        if st.session_state.image_results:
            st.download_button("⬇️ Images Audit (CSV)", pd.DataFrame(st.session_state.image_results).to_csv(index=False), "images_audit.csv", "text/csv", use_container_width=True)

elif not st.session_state.crawl_running:
    st.info("👈 Enter a URL and click **Start Crawl** to begin.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### What it extracts:
        - Page titles, meta descriptions, H1/H2 tags
        - Status codes (200, 301, 404, 500)
        - Canonical URLs, meta robots directives
        - Internal/external link counts
        - Image alt text audit
        - Content analysis (word count, readability)
        - Redirect chains
        - 50+ data points per page
        """)
    with col2:
        st.markdown("""
        ### How it compares to other Screaming spiders:
        - ✅ **Free forever** (Screaming Spider = $259/year)
        - ✅ No install required
        - ✅ Works in your browser
        - ✅ Download results as CSV
        - ✅ Image alt text audit
        - ✅ Readability scoring
        - ⚠️ Some bot-protected sites may block (use desktop version)
        """)

    st.markdown("""
    <div class="footer">
        Built with ❤️ using Python + Streamlit | <a href="https://ko-fi.com/raulvalencia" target="_blank">☕ Support this project</a>
    </div>
    """, unsafe_allow_html=True)
