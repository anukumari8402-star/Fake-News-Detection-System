import streamlit as st
import pickle
import re
import string
import random
from newspaper import Article
import requests
from bs4 import BeautifulSoup

# ── Background images ─────────────────────────────────────────
background_images = [
    "https://eilm.edu.eu/wp-content/uploads/2025/02/human-brain-medical-digital-illustration-2-1200x800.jpg",
    "https://c8.alamy.com/comp/KM0R0Y/drugs-concept-disease-concept-drug-addict-man-with-syringe-using-drugsdark-KM0R0Y.jpg",
    "https://media.gettyimages.com/id/2199601251/video/digital-world-map-background-4k-stock-video.jpg?s=640x640&k=20&c=srL_K-N3pQy_depMJ26Dxtt6gSa02ikzSyNLAyd1j0M=",
    "https://abirpothi.com/wp-content/uploads/2024/05/Courtesy-OpenSea.jpg",
    "https://thumbs.dreamstime.com/b/american-west-antique-pocket-watch-outlaw-gun-38300420.jpg",
    "https://c0.wallpaperflare.com/preview/427/767/687/india-bengaluru-cubbon-park-wire.jpg",
    "https://compote.slate.com/images/dd47a9d7-97bf-4349-b988-874c02bea32c.jpg",
    "https://www.thestatesman.com/wp-content/uploads/2025/02/Untitled-design-2025-02-15T131317.685-jpg.webp",
    "https://static.vecteezy.com/system/resources/thumbnails/071/882/374/small/woman-helping-elderly-person-in-the-rain-photo.jpg",
]

# ── Trusted sources ───────────────────────────────────────────
TRUSTED_SOURCES = [
    "reuters.com", "bbc.com", "bbc.co.uk", "thehindu.com", "indianexpress.com",
    "ndtv.com", "ani.in", "timesofindia.com", "hindustantimes.com", "theprint.in",
    "livemint.com", "moneycontrol.com", "economictimes.indiatimes.com",
    "business-standard.com", "deccanherald.com", "telegraphindia.com",
    "newindianexpress.com", "tribuneindia.com", "outlookindia.com", "news18.com",
    "firstpost.com", "thewire.in", "scroll.in", "thequint.com", "cnn.com",
    "edition.cnn.com", "apnews.com", "nytimes.com", "washingtonpost.com",
    "theguardian.com", "forbes.com", "bloomberg.com", "wsj.com", "npr.org",
    "aljazeera.com", "abcnews.go.com", "cbsnews.com", "nbcnews.com",
    "usatoday.com", "time.com", "dw.com", "france24.com", "indiatoday.in",
    "zeenews.india.com", "jagran.com", "amarujala.com", "bhaskar.com",
    "patrika.com", "livehindustan.com", "navbharattimes.indiatimes.com",
]

# ── Load model artifacts (cached) ────────────────────────────
@st.cache_resource
def load_artifacts():
    m = pickle.load(open("model.pkl",      "rb"))
    v = pickle.load(open("vectorizer.pkl", "rb"))
    a = pickle.load(open("accuracy.pkl",   "rb"))
    return m, v, a

model, vectorizer, accuracy = load_artifacts()

# ── Text cleaning — same as train_model.py ────────────────────
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W',      ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>',   '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\s+',     ' ', text).strip()
    return text

# ── URL → article text (3-layer fallback) ────────────────────
def fetch_text_from_url(url: str):
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    NOISE_PHRASES = [
        "also read", "advertisement", "follow us", "subscribe",
        "breaking news", "click here", "watch live", "download app",
        "read more", "trending now", "related news", "share this",
        "cookie", "privacy policy", "terms of use", "all rights reserved",
        "sign up", "log in", "newsletter", "notification", "sponsored",
    ]

    def clean_fetched(raw: str) -> str:
        lines = raw.split(". ")
        good  = []
        for line in lines:
            l = line.strip().lower()
            if len(l) < 40:
                continue
            if any(noise in l for noise in NOISE_PHRASES):
                continue
            good.append(line.strip())
        return ". ".join(good)

    # Layer 1: newspaper3k
    try:
        art = Article(url)
        art.download()
        art.parse()
        if art.text and len(art.text.strip()) > 300:
            result = clean_fetched(art.text)
            if len(result) > 200:
                return result
    except Exception:
        pass

    # Layer 2: requests + BeautifulSoup
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer",
                          "header", "aside", "form", "iframe",
                          "noscript", "figure", "figcaption"]):
            tag.decompose()

        for tag in soup.find_all(True):
            cls = " ".join(tag.get("class", []))
            tid = tag.get("id", "")
            if any(x in cls.lower() or x in tid.lower()
                   for x in ["ad", "promo", "sidebar", "related",
                              "share", "social", "comment", "widget",
                              "newsletter", "popup", "banner", "footer"]):
                tag.decompose()

        text = ""
        for sel in ["article", "main", '[class*="article"]',
                    '[class*="story"]', '[class*="content"]', "body"]:
            try:
                container = soup.select_one(sel)
            except Exception:
                container = soup.find(sel)

            if container:
                paras = container.find_all("p")
                txt   = " ".join(
                    p.get_text(" ", strip=True)
                    for p in paras
                    if len(p.get_text(strip=True)) > 40
                )
                if len(txt) > 300:
                    text = txt
                    break

        if text:
            result = clean_fetched(text)
            if len(result) > 200:
                return result

    except Exception:
        pass

    # Layer 3: readability
    try:
        from readability import Document
        resp = requests.get(url, headers=HEADERS, timeout=15)
        doc  = Document(resp.text)
        soup = BeautifulSoup(doc.summary(), "html.parser")
        txt  = " ".join(
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 40
        )
        result = clean_fetched(txt)
        if len(result) > 200:
            return result
    except Exception:
        pass

    return None

# ── Core prediction ───────────────────────────────────────────
def predict_news(text: str, url: str = "") -> dict | None:
    cleaned = clean_text(text)
    if len(cleaned.strip()) < 20:
        return None

    vec = vectorizer.transform([cleaned])

    if hasattr(model, "predict_proba"):
        prob      = model.predict_proba(vec)[0]
        classes   = list(model.classes_)
        fake_idx  = classes.index(0)
        real_idx  = classes.index(1)
        fake_pct  = round(prob[fake_idx] * 100, 2)
        real_pct  = round(prob[real_idx] * 100, 2)
    else:
        dec = model.decision_function(vec)[0]
        raw = min(round(abs(dec) * 10, 2), 99.0)
        if dec >= 0:
            real_pct, fake_pct = raw, round(100 - raw, 2)
        else:
            fake_pct, real_pct = raw, round(100 - raw, 2)

    if real_pct >= fake_pct:
        result     = 1
        confidence = real_pct
    else:
        result     = 0
        confidence = fake_pct

    trusted_boost = False
    if url and result == 1:
        for site in TRUSTED_SOURCES:
            if site in url.lower():
                confidence = min(confidence + 10, 99.0)
                real_pct   = min(real_pct   + 10, 99.0)
                trusted_boost = True
                break

    return {
        "result":        result,
        "confidence":    round(confidence, 2),
        "fake_pct":      round(fake_pct,   2),
        "real_pct":      round(real_pct,   2),
        "trusted_boost": trusted_boost,
    }

# ── PAGE CONFIG + CSS ─────────────────────────────────────────
selected_bg = random.choice(background_images)

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown(f"""
<style>
.stApp {{
    background-image: url("{selected_bg}");
    background-size: cover;
    background-position: center;
    animation: zoomBg 20s infinite alternate ease-in-out;
}}
.stApp::before {{
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    backdrop-filter: blur(12px);
    background: rgba(0,0,0,0.55);
    z-index: 0;
}}
@keyframes zoomBg {{
    0%   {{ transform: scale(1);    }}
    100% {{ transform: scale(1.08); }}
}}
section.main > div {{
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 22px;
    padding: 30px 36px;
    margin-top: 20px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
}}
body, h1, h2, h3, h4, label, p, div, span {{
    color: white !important;
}}
textarea, input[type="text"] {{
    border-radius: 12px !important;
    border: 2px solid #7c3aed !important;
    padding: 10px 14px !important;
    background-color: rgba(15,23,42,0.85) !important;
    color: white !important;
    font-size: 0.95rem !important;
}}
textarea::placeholder, input[type="text"]::placeholder {{
    color: #94a3b8 !important;
}}
.stButton > button {{
    background: linear-gradient(135deg, #6d28d9, #a855f7);
    color: white !important;
    border: none;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 600;
    padding: 0.65rem 2.4rem;
    transition: transform 0.2s, box-shadow 0.2s;
    width: 100%;
}}
.stButton > button:hover {{
    transform: scale(1.04);
    box-shadow: 0 0 24px #a855f7aa;
}}
div[data-testid="metric-container"] {{
    background: rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 14px 18px;
    border: 1px solid rgba(255,255,255,0.12);
}}
div[data-testid="metric-container"] label {{
    font-size: 0.8rem !important;
    color: #c4b5fd !important;
}}
div[data-testid="stMetricValue"] {{
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}}
header {{ visibility: hidden; }}
</style>

<h1 style="text-align:center;font-size:2rem;font-weight:800;
           text-shadow:0 0 18px #a855f7;letter-spacing:1px;
           animation:fadeIn 1.5s ease;">
  🧠 AI Fake News Detection System
</h1>
<p style="text-align:center;color:#c4b5fd;margin-top:-6px;font-size:0.9rem;">
  Paste news text <b>or</b> a news link — get instant Real / Fake result
</p>
<style>
@keyframes fadeIn {{
    from {{ opacity:0; transform:translateY(-10px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
</style>
""", unsafe_allow_html=True)

# Accuracy badge
st.markdown(
    f"<p style='text-align:center;'>"
    f"<span style='background:rgba(109,40,217,0.35);border:1px solid #7c3aed;"
    f"border-radius:20px;padding:4px 16px;font-size:0.85rem;color:#e9d5ff;'>"
    f"📊 Model Accuracy: <b>{round(accuracy*100, 2)}%</b>"
    f"</span></p>",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── INPUT FIELDS ──────────────────────────────────────────────
user_input = st.text_area(
    "📝 Paste news article text",
    height=160,
    placeholder="Copy-paste the full news article content here…",
)

url_input = st.text_input(
    "🔗 Or paste a news article URL",
    placeholder="https://www.ndtv.com/india-news/example-article",
)

st.markdown(
    "<p style='color:#94a3b8;font-size:0.78rem;margin-top:-6px;'>"
    "Use text <b>or</b> URL — no need to fill both.</p>",
    unsafe_allow_html=True,
)

# ── SESSION STATE ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("<br>", unsafe_allow_html=True)

# ── PREDICT BUTTON ────────────────────────────────────────────
if st.button("🔍  Detect — Real or Fake?"):

    has_text = bool(user_input.strip())
    has_url  = bool(url_input.strip())

    # Initialise variables — yahi fix hai, pehle ye missing tha
    news_text  = ""
    source_url = ""
    fetched    = None          # ← FIX: hamesha define karo

    if not has_text and not has_url:
        st.warning("⚠️ Please paste news text or a news URL first.")
        st.stop()

    with st.spinner("Analyzing… please wait ⏳"):

        # ── CASE 1: Only text entered (no URL) ───────────────
        if has_text and not has_url:
            news_text = user_input.strip()
            if len(news_text) < 50:
                st.warning("⚠️ Please enter more content — at least a few sentences.")
                st.stop()

        # ── CASE 2: Only URL entered (no text) ───────────────
        elif has_url and not has_text:
            source_url = url_input.strip()
            fetched = fetch_text_from_url(source_url)

            if not fetched or len(fetched.strip()) < 200:
                st.error(
                    "❌ Could not fetch article text from this URL.\n\n"
                    "The site may block bots (paywall / CAPTCHA).\n"
                    "👉 Copy-paste the article text in the text box above."
                )
                st.stop()

            news_text  = fetched
            word_count = len(news_text.split())

            with st.expander(f"📄 Extracted article preview ({word_count} words)"):
                st.write(news_text[:1000] + " …")
                if word_count < 150:
                    st.warning(
                        "⚠️ Only fetched a short snippet. "
                        "For better accuracy, paste the full article text above."
                    )

        # ── CASE 3: Both text AND URL entered ────────────────
        else:
            source_url = url_input.strip()
            fetched    = fetch_text_from_url(source_url)
            word_count = len(fetched.split()) if fetched else 0

            if fetched and word_count >= 150:
                news_text = fetched
                with st.expander(f"📄 Extracted article preview ({word_count} words)"):
                    st.write(fetched[:800] + " …")
            else:
                news_text = user_input.strip()
                if word_count > 0:
                    st.info(
                        f"ℹ️ URL se sirf {word_count} words mile — "
                        "text box wala use kar rahe hain."
                    )

        # ── Run prediction ────────────────────────────────────
        output = predict_news(news_text, url=source_url)

    # ── Prediction output ─────────────────────────────────────
    if output is None:
        st.error("❌ Text is too short after cleaning. Please provide more content.")
        st.stop()

    st.markdown("---")

    if output["result"] == 1:
        st.success(f"✅  **REAL News**  —  Confidence: **{output['confidence']}%**")
        if output["trusted_boost"]:
            st.info("🔒 Trusted source detected — confidence boosted.")
    else:
        st.error(f"🚨  **FAKE News**  —  Confidence: **{output['confidence']}%**")

    col1, col2 = st.columns(2)
    col1.metric("🔴 Fake Probability", f"{output['fake_pct']}%")
    col2.metric("🟢 Real Probability", f"{output['real_pct']}%")

    st.progress(int(min(output["confidence"], 100)))

    if source_url:
        st.markdown(f"🔗 [Read full article]({source_url})")

    # ── History ───────────────────────────────────────────────
    tag = "REAL ✅" if output["result"] == 1 else "FAKE 🚨"
    st.session_state.history.append(
        f"{tag}  |  Fake: {output['fake_pct']}%   Real: {output['real_pct']}%"
    )
    if len(st.session_state.history) > 1:
        st.markdown("#### 📜 Recent Predictions")
        for item in reversed(st.session_state.history[-5:]):
            st.write("•", item)

    # ── Feedback ──────────────────────────────────────────────
    st.markdown("#### 💬 Was this prediction correct?")
    fb = st.radio("", ["—", "Yes ✅", "No ❌"], horizontal=True, key="fb")
    if fb == "Yes ✅":
        st.success("👍 Thanks for the feedback!")
    elif fb == "No ❌":
        st.warning("⚠️ Thanks! We'll keep improving.")


# streamlit run app.py