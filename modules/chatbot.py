import streamlit as st
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, NotFound

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

def ask_llm(context, question):
    prompt = f"""
You are an intelligent data science assistant.

Current project context:
{context}

User question:
{question}

Answer based only on the provided context.
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except NotFound:
        return f"Model '{MODEL_NAME}' is not available for this API key. Try gemini-2.5-pro or gemini-3-flash-preview."
    except ResourceExhausted:
        return "Quota exceeded. Please wait and try again, or check Gemini API billing/quota."