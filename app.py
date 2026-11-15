#!/usr/bin/env python3
"""
SEO Spider - Free Screaming Frog Alternative
Built by Raul @ MaleBasics Corp
"""

import streamlit as st
import pandas as pd
import hmac
import os

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

# Staff-only access. The gate is skipped when SPIDER_PASSWORD is unset so the
# app still runs locally and on the public Streamlit Cloud copy.
_SPIDER_PASSWORD = os.environ.get("SPIDER_PASSWORD", "")
if _SPIDER_PASSWORD and not st.session_state.get("spider_authed"):
    st.markdown('<p class="main-header">🕷️ SEO Spider</p>', unsafe_allow_html=True)
    _pw = st.text_input("Clave de acceso", type="password")
    if _pw:
        if hmac.compare_digest(_pw, _SPIDER_PASSWORD):
            st.session_state.spider_authed = True
            st.rerun()
        st.error("Clave incorrecta")
    st.stop()

if "crawl_running" not in st.session_state:
    st.session_state.crawl_running = False
if "crawl_results" not in st.session_state:
    st.session_state.crawl_results = None
if "image_results" not in st.session_state:
    st.session_state.image_results = None

from spider_core import (  # noqa: E402
    SPIDER_USER_AGENT, parse_header_lines, run_spider,
)

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

    max_pages = st.slider("Max Pages", 10, 50000, 500, step=10,
                          help="No limit when running locally!")
    threads = st.slider("Threads", 1, 20, 2, help="Concurrent requests — lower = safer")
    delay = st.slider("Delay (seconds)", 0.0, 5.0, 1.0, step=0.1,
                      help="Higher delay = less likely to be blocked")

    st.markdown("---")
    st.markdown("**🤖 User-Agent Mode**")
    use_spider_ua = st.toggle(
        "Use SEOSpider User-Agent",
        value=False,
        help="When ON, the crawler identifies itself as 'SEOSpider/1.0' — easy to whitelist in your firewall. When OFF, it mimics a regular browser."
    )
    if use_spider_ua:
        st.code(SPIDER_USER_AGENT, language=None)
        st.caption("Whitelist this User-Agent in your firewall/CDN.")

    st.markdown("---")
    st.markdown("**🔒 Proxy Settings** *(optional — avoids blocks)*")
    proxy_input = st.text_input("Proxy URL", placeholder="http://user:pass@proxy:port",
                                help="Supports HTTP/HTTPS/SOCKS5 proxies. Works with ScraperAPI, Bright Data, Oxylabs, etc.",
                                type="password")
    if proxy_input:
        st.success("Proxy configured", icon="✅")

    st.markdown("---")
    st.markdown("**🔑 Extra request headers** *(optional — Shopify Crawler access)*")
    extra_headers_input = st.text_area(
        "Headers, one per line as `Name: value`",
        placeholder="Signature-Input: sig1=(...)\nSignature: sig1=:...:\nSignature-Agent: \"https://shopify.com\"",
        height=110,
        help="Shopify: Online Store → Preferences → Crawler access → Create signature. Paste the three headers it gives you; they're sent with every request so Shopify stops rate-limiting the crawl.",
    )
    extra_headers = parse_header_lines(extra_headers_input)
    if extra_headers:
        st.success(f"{len(extra_headers)} extra header(s) will be sent", icon="✅")

    st.markdown("---")

    with st.expander("🛡️ Getting blocked? Whitelist the crawler"):
        st.markdown(f"""
**Your crawler User-Agent:**
""")
        st.code(SPIDER_USER_AGENT if use_spider_ua else "Browser UA (rotating)", language=None)
        st.markdown("""
**Tip:** Toggle on "Use SEOSpider User-Agent" above, then whitelist it on your platform:

**Cloudflare:**
1. Go to **Security > WAF > Custom Rules**
2. Create rule: *When User-Agent contains `SEOSpider`* → **Allow**
3. Or go to **Security > Bots** and add the User-Agent to the allow list

**Sucuri:**
1. Go to **Security > Access Control > Whitelist User-Agent**
2. Add `SEOSpider`

**Wordfence (WordPress):**
1. Go to **Wordfence > Firewall > Blocking**
2. Add a whitelist rule for User-Agent containing `SEOSpider`

**Shopify (your own store):**
1. In the store admin go to **Online Store → Preferences → Crawler access → Create signature**
2. Give it a name, the store domain and a validity (max 3 months)
3. Paste the `Signature-Input`, `Signature` and `Signature-Agent` headers it shows into **Extra request headers** above
4. Shopify then serves the full storefront to this crawler without rate limiting

**AWS WAF / CloudFront:**
1. Create a rule in your Web ACL
2. Match on `User-Agent` header containing `SEOSpider` → **Allow**

**Akamai:**
1. Go to **Security > Bot Manager**
2. Add `SEOSpider` to your custom bot allow list

**General / Other WAFs:**
- Whitelist User-Agent: `SEOSpider`
- Or whitelist your IP address in the firewall
""")

    # Donation box
    st.markdown("""
    <div class="donate-box">
        <p style="font-size: 1.1rem; margin-bottom: 8px;">☕ Like this tool?</p>
        <p style="font-size: 0.85rem; margin-bottom: 12px;">Built with ❤️ by a fellow e-commerce operator. If it saves you time (and $$$), consider buying me a coffee!</p>
        <a href="https://ko-fi.com/raulvalencia" target="_blank" style="background: #ffd700; color: #333; padding: 8px 20px; border-radius: 6px; display: inline-block;">☕ Buy Me a Coffee</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚠️ Note:** Some sites with heavy bot protection (Shopify stores with Cloudflare) may block the crawler. Toggle on the SEOSpider User-Agent and whitelist it on your site for best results.")

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

    def _progress(crawled, total, message):
        progress_bar.progress(min(crawled / total, 1.0) if total else 0)
        status_text.text(("🕷️ " if crawled else "⏳ ") + message)

    try:
        data, images, blocked = run_spider(url, max_pages, threads, delay, url_filter[1],
                                           proxy_url=proxy_input.strip() if proxy_input else None,
                                           use_spider_ua=use_spider_ua,
                                           extra_headers=extra_headers or None,
                                           progress=_progress)
        st.session_state.crawl_results = data
        st.session_state.image_results = images
        blocked_msg = f" ({blocked} blocked)" if blocked > 0 else ""
        status_text.text(f"✅ Done! {len(data)} pages crawled{blocked_msg}. {len(images)} images audited.")
        if blocked > 0:
            block_pct = round(blocked / max(len(data), 1) * 100)
            st.warning(
                f"**{blocked} URLs were blocked** (403/429/503) — {block_pct}% of crawled pages. "
                f"Your site's firewall is likely blocking the crawler.\n\n"
                f"**To fix this:** Enable the **SEOSpider User-Agent** in the sidebar, then whitelist "
                f"`SEOSpider` in your firewall/CDN. See the **'Getting blocked?'** section in the sidebar for step-by-step instructions.",
                icon="🚫"
            )
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
    blocked_total = len(df[df["Status Code"].isin([403, 429, 503])])
    if blocked_total > 0:
        c6.metric("Blocked", blocked_total)
    else:
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
        ### How it compares to Screaming Frog:
        - ✅ **Free forever** (Screaming Frog = $259/year)
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
