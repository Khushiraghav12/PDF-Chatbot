import requests
import logging

logger = logging.getLogger(__name__)

def query_ollama_mistral(question, context):
    """
    Query the locally running Ollama Mistral model with RAG-style context.
    """
    system_prompt = "You are a helpful assistant. Use the context from the document to answer the question as clearly and concisely as possible. If the answer is not in the context, say 'Answer not found in the document.'"
    
    prompt = f"""{system_prompt}

Context:
{context}

Question: {question}
Answer:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral:7b-instruct",
                "prompt": prompt,
                "stream": False
            },
            timeout= 300
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        return answer if answer else "⚠️ No answer returned by the model."

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ LLM request failed: {e}")
        return "⚠️ Error contacting the local LLM."

