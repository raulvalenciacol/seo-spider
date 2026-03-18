# SEO Spider — Free Screaming Frog Alternative

A free, browser-based SEO crawler that extracts 50+ data points per page.

## Use it online (no install)

**[Launch SEO Spider](https://seo-spider.streamlit.app)** — runs in your browser, nothing to download.

---

## Install locally (unlimited pages, faster)

### Windows (one-click)

1. **[Download ZIP](https://github.com/raulvalenciacol/seo-spider/archive/refs/heads/main.zip)** and extract it anywhere
2. Double-click **`install_and_run.bat`**
3. That's it — it installs Python + dependencies automatically and opens the app

**Or clone with Git** (recommended — gets automatic updates):
```
git clone https://github.com/raulvalenciacol/seo-spider.git
cd seo-spider
install_and_run.bat
```

### Mac / Linux (one-click)

```bash
git clone https://github.com/raulvalenciacol/seo-spider.git
cd seo-spider
chmod +x install_and_run.sh
./install_and_run.sh
```

Or without Git:
1. **[Download ZIP](https://github.com/raulvalenciacol/seo-spider/archive/refs/heads/main.zip)** and extract it
2. Open Terminal in the extracted folder
3. Run: `chmod +x install_and_run.sh && ./install_and_run.sh`

---

## Automatic updates

If you installed via `git clone`, the app **checks for updates every time you run it**. You'll always have the latest features and fixes — just run the same script again.

If you installed via ZIP download, re-download the ZIP to get the latest version.

---

## What it extracts

- Page titles, meta descriptions, H1/H2 tags
- Status codes (200, 301, 404, 500)
- Canonical URLs, meta robots directives
- Internal/external link counts
- Image alt text audit
- Content analysis (word count, readability scoring)
- Redirect chains
- 50+ data points per page
- Export everything to CSV

## How it compares

| Feature | SEO Spider | Screaming Frog |
|---------|-----------|----------------|
| Price | **Free** | $259/year |
| Install required | No | Yes |
| Works in browser | Yes | No |
| CSV export | Yes | Yes |
| Image audit | Yes | Yes |
| Max pages (free) | Unlimited (local) | 500 |

## Built with

Python, Streamlit, BeautifulSoup, Pandas

---

[Buy me a coffee](https://ko-fi.com/raulvalencia) if this saved you time!
