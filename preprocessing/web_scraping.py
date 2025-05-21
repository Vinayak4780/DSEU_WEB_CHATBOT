# import re
# import zipfile
# from pathlib import Path
# import base64

# # ──────────────────────────────────────────────────────────────────────────────
# # CONFIGURATION
# ZIP_PATH   = Path("constants(8).zip")  # path to your ZIP
# OUT_PATH   = Path("all_files_content.txt")      # aggregated output
# # ──────────────────────────────────────────────────────────────────────────────

# with zipfile.ZipFile(ZIP_PATH, "r") as zf, OUT_PATH.open("w", encoding="utf-8") as out:
#     for info in zf.infolist():
#         if info.is_dir():
#             continue

#         # Read raw bytes
#         data = zf.read(info)

#         # Try to decode as UTF-8
#         try:
#             text = data.decode("utf-8")
#             mode = "utf-8"
#         except UnicodeDecodeError:
#             # Fallback: Base64-encode the bytes
#             text = base64.b64encode(data).decode("ascii")
#             mode = "base64"

#         # Write header and content
#         out.write(f"=== File: {info.filename}  (decoded as {mode}) ===\n")
#         out.write(text)
#         out.write("\n\n")

# print(f"All files extracted → {OUT_PATH.resolve()}")

import re
import nltk

# ─── 0) If running first time, uncomment these: ────────────────────────────
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# # ─── Prepare NLP tools ─────────────────────────────────────────────────────
# stop_words   = set(stopwords.words('english'))
# lemmatizer   = WordNetLemmatizer()
# word_pattern = re.compile(r'\w+')   # matches letters & digits sequences

# # ─── 1) Read your JS‐in‐.txt file ──────────────────────────────────────────
# with open('/home/ubuntu/Desktop/mongo_db_chatbot/all_files_content.txt', 'r', encoding='utf-8') as f:
#     js_code = f.read()

# # ─── 2) Pull out comments & string-literals ───────────────────────────────
# single_line = re.findall(r'//(.*)', js_code)
# multi_line  = re.findall(r'/\*([\s\S]*?)\*/', js_code)
# dq_strings  = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', js_code)
# sq_strings  = re.findall(r"'([^'\\]*(?:\\.[^'\\]*)*)'", js_code)

# blocks = single_line + multi_line + dq_strings + sq_strings

# # ─── 3→7) Clean & preprocess each block ──────────────────────────────────
# processed_lines = []
# for blk in blocks:
#     # 3) lowercase
#     blk = blk.lower()
#     # 4) remove any punctuation except letters/digits/whitespace
#     blk = re.sub(r'[^a-z0-9\s]', ' ', blk)
#     # collapse runs of whitespace
#     blk = re.sub(r'\s+', ' ', blk).strip()
#     if not blk:
#         continue

#     # 5) tokenize on word characters (keeps digits)
#     tokens = word_pattern.findall(blk)

#     # 6) remove stop-words but keep any token containing a digit
#     tokens = [
#         t for t in tokens
#         if (t in stop_words) == False or re.search(r'\d', t)
#     ]

#     # 7) lemmatize each token
#     tokens = [lemmatizer.lemmatize(t) for t in tokens]

#     # rejoin and store
#     processed_lines.append(' '.join(tokens))

# # ─── 8) Write to output.txt ───────────────────────────────────────────────
# with open('output.txt', 'w', encoding='utf-8') as out:
#     out.write('\n'.join(processed_lines))

# print(f"Done → extracted and preprocessed {len(processed_lines)} lines.")
import os, time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

# ─── CONFIG ────────────────────────────────────────────────────────────────
INPUT_FILE    = "output.txt"  # path to your text file
INDEX_DIR     = "data/web_scraping"
MODEL_PATH    = "models/your-ggml-model.bin"  # e.g. a small 7B llama-cpp quant
CTX_CHUNK_SZ  = 500
CTX_OVERLAP   = 0
EMB_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"

# ─── STEP 1: Read & chunk your text ────────────────────────────────────────
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CTX_CHUNK_SZ,
    chunk_overlap=CTX_OVERLAP
)
texts = splitter.split_text(full_text)

# ─── STEP 2: Build & save FAISS index (run once) ──────────────────────────
if not os.path.isdir(INDEX_DIR):
    os.makedirs(INDEX_DIR)
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    faiss_index = FAISS.from_texts(texts, embeddings)
    faiss_index.save_local(INDEX_DIR)
    print(f"✅ Built and saved index with {len(texts)} chunks.")
else:
    print("ℹ️  Index folder exists; skipping build.")

