
# from flask import Flask, request, jsonify, render_template, redirect, url_for
# from flask_cors import CORS
# import os
# import uuid
# import logging
# import io
# import time

# import fitz  # PyMuPDF
# import pytesseract
# from PIL import Image
# import tabula
# import pandas as pd

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # Ollama LLM wrapper
# from llm_mistral import query_ollama_mistral

# app = Flask(__name__)
# CORS(app)

# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# embedding_model = SentenceTransformer('./models/all-MiniLM-L6-v1')

# document_chunks = []
# chunk_embeddings = []

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# # 📌 TEXT + IMAGE (OCR) extraction
# def extract_text_and_images(pdf_path):
#     chunks = []
#     doc = fitz.open(pdf_path)
#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)

#         # Extract plain text
#         text = page.get_text("text")
#         if text.strip():
#             chunks.append({
#                 'type': 'text',
#                 'content': text.strip(),
#                 'source': f'Page {page_num + 1}'
#             })

#         # Extract images + OCR
#         images = page.get_images(full=True)
#         for img_index, img in enumerate(images):
#             xref = img[0]
#             base_image = doc.extract_image(xref)
#             image_bytes = base_image["image"]
#             image = Image.open(io.BytesIO(image_bytes))
#             ocr_text = pytesseract.image_to_string(image)
#             if ocr_text.strip():
#                 chunks.append({
#                     'type': 'image',
#                     'content': ocr_text.strip(),
#                     'source': f'Image on Page {page_num + 1}'
#                 })

#     doc.close()
#     return chunks


# # 📌 Table extraction
# def extract_tables(pdf_path):
#     chunks = []
#     try:
#         tables = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True, encoding="ISO-8859-1")
#         for i, table in enumerate(tables):
#             if isinstance(table, pd.DataFrame) and not table.empty:
#                 text = table.to_string()
#                 chunks.append({
#                     'type': 'table',
#                     'content': text,
#                     'source': f'Table {i + 1}'
#                 })
#     except Exception as e:
#         logger.warning(f"❌ Table extraction error: {e}")
#     return chunks


# # 📌 Embedding function
# def embed_chunks(chunks):
#     texts = [chunk['content'] for chunk in chunks]
#     embeddings = embedding_model.encode(texts, show_progress_bar=True)
#     return embeddings


# @app.route('/')
# def index():
#     return render_template('index.html')


# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'pdf' not in request.files:
#         return "No PDF uploaded", 400

#     file = request.files['pdf']
#     filename = f"{uuid.uuid4()}.pdf"
#     filepath = os.path.join(UPLOAD_FOLDER, filename)
#     file.save(filepath)

#     global document_chunks, chunk_embeddings
#     document_chunks.clear()

#     # Extract text, images, tables
#     text_chunks = extract_text_and_images(filepath)
#     table_chunks = extract_tables(filepath)
#     all_chunks = text_chunks + table_chunks

#     if not all_chunks:
#         return "❌ No content extracted from PDF", 500

#     embeddings = embed_chunks(all_chunks)

#     document_chunks = all_chunks
#     chunk_embeddings = embeddings

#     # Log separate counts
#     num_text = len([c for c in all_chunks if c['type'] == 'text'])
#     num_table = len([c for c in all_chunks if c['type'] == 'table'])
#     num_image = len([c for c in all_chunks if c['type'] == 'image'])
#     print(f"✅ Extracted {len(all_chunks)} chunks → Text:{num_text}, Tables:{num_table}, Images:{num_image}")

#     return redirect(url_for('chat'))


# @app.route('/chat')
# def chat():
#     return render_template('chat.html')


# @app.route('/ask', methods=['POST'])
# def ask():
#     """
#     Accepts a question, retrieves top chunks, and calls Ollama (Mistral).
#     Optimized for speed.
#     """
#     data = request.get_json()
#     question = data.get('question', '').strip()

#     if not question or not document_chunks:
#         return jsonify({'answer': "❌ No data available or no question provided."})

#     start_time = time.time()

#     # Find top 3 relevant chunks
#     query_embedding = embedding_model.encode([question])[0]
#     similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
#     top_indices = similarities.argsort()[-3:][::-1]
#     top_chunks = [document_chunks[i] for i in top_indices]

#     # Limit chunk size (smaller for speed)
#     context_parts = []
#     for chunk in top_chunks:
#         max_chars = 1000 if chunk['type'] == 'text' else 600
#         context_parts.append(f"[{chunk['type'].upper()} - {chunk['source']}]\n{chunk['content'][:max_chars]}")

#     context = "\n\n".join(context_parts)

#     print(f"📥 Question: {question}")
#     print(f"📊 Top similarity score: {similarities[top_indices[0]]:.4f}")

#     # 🔥 Send to Ollama (Mistral)
#     answer = query_ollama_mistral(question, context)

#     elapsed = time.time() - start_time
#     print(f"⚡ Answer generated in {elapsed:.2f} sec")

#     return jsonify({'answer': answer})


# if __name__ == '__main__':
#     print("✅ Flask is launching...")
#     app.run(debug=True)
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import os, uuid, io, logging, time
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import camelot  # for table extraction
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Ollama LLM wrapper
from llm_mistral import query_ollama_mistral

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

embedding_model = SentenceTransformer('./models/all-MiniLM-L6-v1')

document_chunks = []
chunk_embeddings = []
table_data = []  # structured table list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 📌 TEXT + IMAGE (OCR) extraction
def extract_text_and_images(pdf_path):
    chunks = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Extract plain text
        text = page.get_text("text")
        if text.strip():
            chunks.append({
                'type': 'text',
                'content': text.strip(),
                'source': f'Page {page_num + 1}'
            })

        # Extract images + OCR
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(image)
            if ocr_text.strip():
                chunks.append({
                    'type': 'image',
                    'content': ocr_text.strip(),
                    'source': f'Image on Page {page_num + 1}'
                })
    doc.close()
    return chunks

# 📌 Table extraction using Camelot
def extract_tables(pdf_path):
    chunks = []
    global table_data
    table_data.clear()
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        for i, table in enumerate(tables):
            df = table.df
            # Skip empty tables
            if df.empty: 
                continue
            chunks.append({
                'type': 'table',
                'content': df.to_string(),
                'source': f'Table {i + 1}'
            })
            # Convert table rows to structured data
            # Assuming table has columns: "Area (sq.m.)", "HRR"
            for idx, row in df.iterrows():
                try:
                    area_range = row[0].replace("–", "-").split("-")
                    min_area = int(area_range[0].strip())
                    max_area = int(area_range[1].strip())
                    hrr = float(row[1].strip())
                    table_data.append({
                        'min_area': min_area,
                        'max_area': max_area,
                        'hrr': hrr
                    })
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"❌ Table extraction error: {e}")
    return chunks

# 📌 Embedding function
def embed_chunks(chunks):
    texts = [chunk['content'] for chunk in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    return embeddings

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf' not in request.files:
        return "No PDF uploaded", 400

    file = request.files['pdf']
    filename = f"{uuid.uuid4()}.pdf"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    global document_chunks, chunk_embeddings
    document_chunks.clear()

    # Extract text, images, tables
    text_chunks = extract_text_and_images(filepath)
    table_chunks = extract_tables(filepath)
    all_chunks = text_chunks + table_chunks

    if not all_chunks:
        return "❌ No content extracted from PDF", 500

    embeddings = embed_chunks(all_chunks)

    document_chunks = all_chunks
    chunk_embeddings = embeddings

    # Log counts
    num_text = len([c for c in all_chunks if c['type'] == 'text'])
    num_table = len([c for c in all_chunks if c['type'] == 'table'])
    num_image = len([c for c in all_chunks if c['type'] == 'image'])
    print(f"✅ Extracted {len(all_chunks)} chunks → Text:{num_text}, Tables:{num_table}, Images:{num_image}")

    return redirect(url_for('chat'))

@app.route('/chat')
def chat():
    return render_template('chat.html')

# 📌 HRR numeric lookup
def find_hrr_for_area(area):
    if not table_data:
        return None
    # Check exact range
    for row in table_data:
        if row['min_area'] <= area <= row['max_area']:
            return row['hrr']
    # Otherwise pick closest lower range
    lower_rows = [r for r in table_data if r['max_area'] < area]
    if lower_rows:
        closest = max(lower_rows, key=lambda x: x['max_area'])
        return closest['hrr']
    return None

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '').strip()

    if not question or not document_chunks:
        return jsonify({'answer': "❌ No data available or no question provided."})

    start_time = time.time()

    # If question is about area and HRR, try numeric table lookup
    import re
    area_match = re.search(r'(\d+)[–-](\d+)\s*sq\.? meters?', question)
    if area_match:
        area = int((int(area_match.group(1)) + int(area_match.group(2))) / 2)  # take midpoint
        hrr_value = find_hrr_for_area(area)
        if hrr_value is not None:
            answer = f"From table: The HRR for workmen living in {area_match.group(1)}–{area_match.group(2)} sq. meters is approximately Rs. {hrr_value:.0f} per month."
            return jsonify({'answer': answer})

    # Otherwise, use semantic search + Ollama LLM
    query_embedding = embedding_model.encode([question])[0]
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    top_indices = similarities.argsort()[-3:][::-1]
    top_chunks = [document_chunks[i] for i in top_indices]

    context_parts = []
    for chunk in top_chunks:
        max_chars = 1000 if chunk['type'] == 'text' else 600
        context_parts.append(f"[{chunk['type'].upper()} - {chunk['source']}]\n{chunk['content'][:max_chars]}")
    context = "\n\n".join(context_parts)

    print(f"📥 Question: {question}")
    print(f"📊 Top similarity score: {similarities[top_indices[0]]:.4f}")

    answer = query_ollama_mistral(question, context)
    elapsed = time.time() - start_time
    print(f"⚡ Answer generated in {elapsed:.2f} sec")

    return jsonify({'answer': answer})

if __name__ == '__main__':
    print("✅ Flask is launching...")
    app.run(debug=True)
