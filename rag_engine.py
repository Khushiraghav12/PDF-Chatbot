# import fitz  # PyMuPDF
# import pytesseract
# from PIL import Image
# import io
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import os
# import cv2

# embedding_model = SentenceTransformer('./models/all-MiniLM-L6-v1')  # Local model

# def extract_chunks_from_pdf(pdf_path):
#     chunks = []

#     # Load PDF
#     doc = fitz.open(pdf_path)
#     for i in range(len(doc)):
#         page = doc.load_page(i)
        
#         # TEXT
#         text = page.get_text("text")
#         if text.strip():
#             chunks.append({
#                 "type": "text",
#                 "content": text,
#                 "page": i + 1
#             })

#         # IMAGE (OCR)
#         for img_index, img in enumerate(page.get_images(full=True)):
#             base_image = doc.extract_image(img[0])
#             image_bytes = base_image["image"]
#             image = Image.open(io.BytesIO(image_bytes))
#             ocr_text = pytesseract.image_to_string(image)
#             if ocr_text.strip():
#                 chunks.append({
#                     "type": "image",
#                     "content": ocr_text,
#                     "page": i + 1
#                 })

#     doc.close()
#     return chunks


# def generate_embeddings(chunks):
#     texts = [chunk["content"] for chunk in chunks]
#     embeddings = embedding_model.encode(texts)
#     return embeddings


# def answer_question(question, chunks, embeddings):
#     q_embed = embedding_model.encode([question])
#     scores = cosine_similarity(q_embed, embeddings)[0]
#     top_idx = np.argmax(scores)
#     top_chunk = chunks[top_idx]

#     return {
#         "answer": top_chunk["content"],
#         "page": top_chunk["page"],
#         "type": top_chunk["type"],
#         "score": scores[top_idx]
#     }
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticSearchEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = []
        self.chunks = []

    def index_chunks(self, chunks):
        self.chunks = chunks
        texts = [chunk['content'] for chunk in chunks]
        self.embeddings = self.model.encode(texts, convert_to_tensor=True)

    def search(self, query, top_k=5):
        if not self.embeddings:
            return []

        query_embedding = self.model.encode([query], convert_to_tensor=True)
        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return [self.chunks[i]['content'] for i in top_indices]
