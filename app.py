# app.py
import os
import io
import json
import uuid
import random
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()  # if you keep a .env with OPENAI_API_KEY locally

# -------- CONFIG ----------
DATA_DIR = Path("data")
UPLOAD_DIR = Path("images/uploads")
BASKET_FILE = DATA_DIR / "basket.json"
BASKET_BG = "images/basket_bg.png"  # optional background (place a file here to use)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
if not BASKET_FILE.exists():
    BASKET_FILE.write_text("[]")

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TEXT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

# ---------- Utilities ----------
def load_basket():
    try:
        return json.loads(BASKET_FILE.read_text())
    except Exception:
        return []

def save_basket(basket):
    BASKET_FILE.write_text(json.dumps(basket, indent=2))

def save_uploaded_file(uploaded_file):
    # save to images/uploads with a unique name
    ext = Path(uploaded_file.name).suffix.lower()
    uid = uuid.uuid4().hex[:12]
    filename = UPLOAD_DIR / f"{uid}{ext}"
    with open(filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(filename)

def load_image(path_or_buffer):
    if isinstance(path_or_buffer, (str, Path)):
        return Image.open(path_or_buffer).convert("RGB")
    else:
        return Image.open(io.BytesIO(path_or_buffer)).convert("RGB")

def get_dominant_colors(image: Image.Image, k=3):
    small = image.resize((150, 150))
    arr = np.array(small).reshape(-1, 3).astype(float)
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(arr)
    centers = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    total = counts.sum()
    order = np.argsort(-counts)
    colors = []
    for idx in order:
        c = centers[idx]
        pct = counts[idx] / total
        hexc = '#%02x%02x%02x' % tuple(c)
        colors.append({"hex": hexc, "pct": float(pct)})
    return colors

def edge_density(image: Image.Image):
    gray = ImageOps.grayscale(image).resize((150, 150))
    arr = np.array(gray).astype(float) / 255.0
    gy, gx = np.gradient(arr)
    mag = np.sqrt(gx**2 + gy**2)
    return float(np.mean(mag))

def build_feature_summary(colors, edge_d):
    main = colors[0]
    color_desc = f"{main['hex']} ({int(main['pct']*100)}%)"
    other = ""
    if len(colors) > 1:
        other = ", plus " + ", ".join([c["hex"] for c in colors[1:]])
    texture = "textured" if edge_d > 0.06 else "smooth"
    return f"{color_desc}{other}; vibe: {texture} (edge={edge_d:.3f})"

def get_embedding(text: str):
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(resp.data[0].embedding, dtype=float)

def generate_funny_line(context: str):
    # small chat call to produce a funny one-liner; keep short
    prompt = f"Write one short (<=20 words), funny, family-friendly line about a sock {context}."
    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role":"user","content": prompt}],
        max_tokens=40,
        temperature=0.9,
    )
    return resp.choices[0].message.content.strip()

# ---------- Matching ----------
SIMILARITY_THRESHOLD = 0.82  # tune this: lower -> more matches; higher -> stricter

def find_best_match(emb_new, basket):
    if not basket:
        return None, 0.0
    best_idx = None
    best_score = -1.0
    for i, item in enumerate(basket):
        emb_old = np.array(item["embedding"], dtype=float)
        score = float(cosine_similarity([emb_new],[emb_old])[0][0])
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx, best_score

# ---------- Streamlit UI ----------
st.set_page_config(page_title="SockMatch 2.0 🧦", layout="wide")
st.title("SockMatch 2.0 — the Lost & Found Laundry Assistant")

st.markdown("""
Upload one or more lonely socks. The app checks the Basket of singles and either:
- suggests a likely match (you confirm), or
- adds the new socks to the basket with a funny line.
""")

uploaded = st.file_uploader("Upload one or many socks 🧦", type=["png","jpg","jpeg"], accept_multiple_files=True)

basket = load_basket()  # list of dicts

# Left column: uploads & processing; Right column: visual basket
left, right = st.columns([1, 1])

with left:
    if uploaded:
        st.markdown(f"### Processing {len(uploaded)} uploaded file(s)...")
        for up in uploaded:
            try:
                # save uploaded file
                saved_path = save_uploaded_file(up)
                img = load_image(saved_path)

                # analyze
                colors = get_dominant_colors(img, k=3)
                ed = edge_density(img)
                feat = build_feature_summary(colors, ed)
                desc_text = f"{feat}"

                # embedding
                emb = get_embedding(desc_text)
                emb_list = emb.tolist()

                # find best match in basket
                idx, score = find_best_match(emb, basket)

                if idx is not None and score >= SIMILARITY_THRESHOLD:
                    candidate = basket[idx]
                    st.markdown("**Possible match found:**")
                    cols = st.columns([1,1])
                    with cols[0]:
                        st.image(saved_path, caption="New sock", width=200)
                        st.write(desc_text)
                    with cols[1]:
                        st.image(candidate["image_path"], caption=f"Basket sock (since {candidate['upload_date'][:10]})", width=200)
                        st.write(candidate["summary"])

                    # Ask for confirmation
                    confirm_col, skip_col = st.columns(2)
                    with confirm_col:
                        if st.button(f"✅ Yes, match with {Path(candidate['image_path']).name}", key=f"confirm_{saved_path}"):
                            # compute days apart
                            old_date = datetime.fromisoformat(candidate["upload_date"])
                            days_apart = (datetime.now() - old_date).days
                            # Remove old sock
                            removed = basket.pop(idx)
                            save_basket(basket)
                            # fun line
                            line = generate_funny_line(f"reuniting after {days_apart} days apart.")
                            st.success(f"💞 {line}")
                            st.balloons()
                            continue  # proceed to next uploaded sock
                    with skip_col:
                        if st.button("❌ Not a match", key=f"skip_{saved_path}"):
                            # treat it as no-match: add new sock into basket
                            new_entry = {
                                "id": uuid.uuid4().hex,
                                "image_path": saved_path,
                                "summary": desc_text,
                                "embedding": emb_list,
                                "upload_date": datetime.now().isoformat()
                            }
                            basket.append(new_entry)
                            save_basket(basket)
                            line = generate_funny_line("joining the lonely basket, single and hopeful")
                            st.info(line)
                            continue

                else:
                    # No candidate or below threshold
                    new_entry = {
                        "id": uuid.uuid4().hex,
                        "image_path": saved_path,
                        "summary": desc_text,
                        "embedding": emb_list,
                        "upload_date": datetime.now().isoformat()
                    }
                    basket.append(new_entry)
                    save_basket(basket)
                    # fun message when no matches
                    line = generate_funny_line("added to the singles basket, waiting for a soulmate")
                    st.info(line)

            except Exception as e:
                st.error(f"Error processing {up.name}: {e}")

with right:
    st.markdown("### 🧺 Visual Lonely Sock Basket")
    # show basket background if available
    if Path(BASKET_BG).exists():
        # build overlay HTML to place sock thumbnails randomly over the basket background
        html = "<div style='position:relative;width:100%;max-width:900px;'>"
        html += f"<img src='/{BASKET_BG}' style='width:100%;display:block;'/>"
        # place each sock
        for i, item in enumerate(basket):
            left_pct = random.randint(8, 80)
            top_pct = random.randint(30, 75)
            html += (
                f"<img src='/{item['image_path']}' "
                f"style='position:absolute;left:{left_pct}%;top:{top_pct}%;width:80px;border-radius:8px;border:2px solid rgba(255,255,255,0.9);box-shadow:0 4px 8px rgba(0,0,0,0.2);'/>"
            )
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
    else:
        # simple grid fallback
        if not basket:
            st.info("Basket is empty — add some lonely socks!")
        else:
            cols = st.columns(5)
            for i, item in enumerate(basket):
                with cols[i % 5]:
                    st.image(item["image_path"], width=120, caption=f"{i+1}. {item['upload_date'][:10]}")

# small footer / stats
st.markdown("---")
st.write(f"Basket size: {len(basket)} socks")
